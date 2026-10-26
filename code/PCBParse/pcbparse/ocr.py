"""PP-OCRv5 text recognition via onnxruntime (no paddle dependency)."""
import os
import math
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.environ.get('PCB_OCR_MODELS', os.path.join(os.path.dirname(_HERE), 'models', 'ppocrv5'))


OCR_BATCH_PIX = 16 * 2560     # batch x padded width above which crops are recognised one by one
OCR_MAX_W = 4096               # widest input to the recogniser (longer crops are squashed)
OCR_OWN_WIDTH = os.environ.get('PCB_OCR_OWN_WIDTH', '0') == '1'   # A/B pending
OCR_MIN_W = int(os.environ.get('PCB_OCR_MIN_W', '0'))    # PaddleOCR pads short crops to 320


class Recognizer:
    def __init__(self, model_dir=_MODEL_DIR, use_gpu=False):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get('PCB_OCR_THREADS', '0')) or 0
        prov = ['CPUExecutionProvider']
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            prov = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(os.path.join(model_dir, 'rec.onnx'), so, providers=prov)
        self.inp = self.sess.get_inputs()[0].name
        chars = []
        with open(os.path.join(model_dir, 'ppocrv5_dict.txt'), encoding='utf-8') as f:
            for line in f:
                chars.append(line.rstrip('\n').rstrip('\r'))
        chars.append(' ')
        self.chars = ['<blank>'] + chars
        self.H = 48
        self.cache = {}

    def _prep(self, img, W):
        h, w = img.shape[:2]
        ratio = w / float(h)
        rw = min(W, int(math.ceil(self.H * ratio)))
        rw = max(rw, 8)
        im = cv2.resize(img, (rw, self.H), interpolation=cv2.INTER_CUBIC if h < self.H else cv2.INTER_AREA)
        im = im.astype(np.float32).transpose(2, 0, 1) / 255.0
        im = (im - 0.5) / 0.5
        out = np.zeros((3, self.H, W), np.float32)
        out[:, :, :rw] = im
        return out

    def decode(self, probs):
        idx = probs.argmax(-1)
        conf = probs.max(-1)
        res = []
        confs = []
        positions = []
        prev = -1
        for t, (i, c) in enumerate(zip(idx, conf)):
            if i != prev and i != 0:
                res.append(self.chars[i] if i < len(self.chars) else '')
                confs.append(float(c))
                positions.append(t)
            prev = i
        text = ''.join(res)
        return text, (float(np.mean(confs)) if confs else 0.0), confs, positions

    def constrained(self, probs, allowed):
        """Greedy CTC decode restricted to an allowed character set."""
        if not hasattr(self, '_idx_cache'):
            self._idx_cache = {}
        key = allowed
        if key not in self._idx_cache:
            ids = [0] + [i for i, c in enumerate(self.chars) if c in allowed and i > 0]
            self._idx_cache[key] = np.array(ids)
        ids = self._idx_cache[key]
        sub = probs[:, ids]
        idx = sub.argmax(-1)
        res = []
        confs = []
        prev = -1
        for t, i in enumerate(idx):
            if i != prev and i != 0:
                res.append(self.chars[ids[i]])
                confs.append(float(sub[t, i]))
            prev = i
        return ''.join(res), (float(np.mean(confs)) if confs else 0.0)

    def decode_split(self, probs, set_a, set_b, k):
        """Greedy CTC decode: first k emitted characters from set_a, the rest from set_b."""
        ia = np.array([i for i, c in enumerate(self.chars) if c in set_a and i > 0])
        ib = np.array([i for i, c in enumerate(self.chars) if c in set_b and i > 0])
        out = []
        confs = []
        prev = -1
        for t in range(probs.shape[0]):
            pr = probs[t]
            ids = ia if len(out) < k else ib
            j = ids[np.argmax(pr[ids])]
            if pr[0] >= pr[j]:
                prev = 0
                continue
            if j != prev:
                out.append(self.chars[j])
                confs.append(float(pr[j]))
            prev = j
        return ''.join(out), (float(np.mean(confs)) if confs else 0.0)

    def __call__(self, imgs, batch=16, return_chars=False, return_probs=False):
        """imgs: list of RGB uint8 crops. Returns list of (text, conf)."""
        out = [None] * len(imgs)
        order = sorted(range(len(imgs)), key=lambda i: imgs[i].shape[1] / max(imgs[i].shape[0], 1))
        groups = []
        if OCR_OWN_WIDTH:
            # every crop padded to its own width (crops of one width share a batch): a read
            # never depends on which other texts happen to be on the page
            byw = {}
            for i in order:
                r = imgs[i].shape[1] / max(imgs[i].shape[0], 1)
                Wi = int(math.ceil((int(max(self.H * r, 32)) + 8) / 8) * 8)
                Wi = max(Wi, OCR_MIN_W)
                byw.setdefault(min(Wi, OCR_MAX_W), []).append(i)
            for Wi, ids in sorted(byw.items()):
                step = max(1, min(batch * 4, OCR_BATCH_PIX // Wi))
                for k in range(0, len(ids), step):
                    groups.append((ids[k:k + step], Wi))
            order = []
        for k in range(0, len(order), batch):
            ids = order[k:k + batch]
            maxr = max(imgs[i].shape[1] / max(imgs[i].shape[0], 1) for i in ids)
            W = int(max(self.H * maxr, 32)) + 8
            W = int(math.ceil(W / 8) * 8)
            if len(ids) * W <= OCR_BATCH_PIX:
                groups.append((ids, W))
            else:
                # very long 'lines' (a mis-grouped row of marks): one at a time, width capped,
                # so that a single page cannot exhaust memory in the recogniser
                for i in ids:
                    r = imgs[i].shape[1] / max(imgs[i].shape[0], 1)
                    Wi = int(math.ceil((int(max(self.H * r, 32)) + 8) / 8) * 8)
                    groups.append(([i], min(Wi, OCR_MAX_W)))
        for ids, W in groups:
            x = np.stack([self._prep(imgs[i], W) for i in ids])
            probs = self.sess.run(None, {self.inp: x})[0]
            for j, i in enumerate(ids):
                t, c, cc, pos = self.decode(probs[j])
                if return_probs:
                    # trim padding columns: keep all (blank-dominated padding is harmless)
                    out[i] = (t, c, probs[j])
                else:
                    out[i] = (t, c, cc) if return_chars else (t, c)
        return out


_REC = None


def get_rec():
    global _REC
    if _REC is None:
        _REC = Recognizer()
    return _REC
