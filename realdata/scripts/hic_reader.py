#!/usr/bin/env python3
"""Minimal .hic reader (format v8, Rao et al. GEO files) over HTTP range requests.
Implements the subset of the Juicebox DatasetReaderV2 spec needed for:
intra-chromosomal observed matrices at one BP resolution + KR/VC norm vectors.
Written from the Juicebox source (DatasetReaderV2.java, BinReader.java)."""
import struct, zlib, io, time
import requests
import numpy as np

class Remote:
    def __init__(self, url):
        self.url = url
        self.s = requests.Session()
    def read(self, pos, n):
        if n <= 0:
            return b''
        last = None
        for attempt in range(6):
            try:
                r = self.s.get(self.url, headers={'Range': f'bytes={pos}-{pos+n-1}'}, timeout=300)
                r.raise_for_status()
                assert len(r.content) == n, f'range read short: got {len(r.content)} want {n} at {pos}'
                return r.content
            except Exception as e:  # NCBI range fetches occasionally time out; retry with backoff
                last = e
                time.sleep(5 * (attempt + 1))
        raise last

class Cursor:
    def __init__(self, buf):
        self.b = buf; self.o = 0
    def read(self, n):
        d = self.b[self.o:self.o+n]; self.o += n; return d
    def cstr(self):
        e = self.b.index(b'\x00', self.o)
        s = self.b[self.o:e].decode(); self.o = e + 1; return s
    def i32(self): v, = struct.unpack_from('<i', self.b, self.o); self.o += 4; return v
    def i64(self): v, = struct.unpack_from('<q', self.b, self.o); self.o += 8; return v
    def f32(self): v, = struct.unpack_from('<f', self.b, self.o); self.o += 4; return v
    def f64(self): v, = struct.unpack_from('<d', self.b, self.o); self.o += 8; return v
    def i16(self): v, = struct.unpack_from('<h', self.b, self.o); self.o += 2; return v
    def u8(self): v = self.b[self.o]; self.o += 1; return v

class HicFile:
    def __init__(self, url):
        self.r = Remote(url)
        head = Cursor(self.r.read(0, 1 << 20))
        assert head.cstr() == 'HIC'
        self.version = head.i32()
        assert self.version in (7, 8), f'expected v7/v8, got {self.version}'
        self.master_pos = head.i64()
        self.genome = head.cstr()
        self.attributes = {}
        for _ in range(head.i32()):
            k = head.cstr(); self.attributes[k] = head.cstr()
        self.chroms = []          # (name, length)
        self.chr_index = {}
        for i in range(head.i32()):
            nm = head.cstr(); ln = head.i32()
            self.chr_index[nm] = i
            self.chroms.append((nm, ln))
        self.bp_res = [head.i32() for _ in range(head.i32())]
        self.frag_res = [head.i32() for _ in range(head.i32())]
        self._footer()

    def _footer(self):
        # master index
        buf = self.r.read(self.master_pos, 4)
        nbytes, = struct.unpack('<i', buf)
        body = Cursor(self.r.read(self.master_pos + 4, nbytes))
        self.master = {}
        for _ in range(body.i32()):
            key = body.cstr(); pos = body.i64(); size = body.i32()
            self.master[key] = (pos, size)
        # expected values (NONE): parse through to find norm-vector section start
        n_exp = body.i32()
        for _ in range(n_exp):
            body.cstr(); body.i32()                    # unit, binSize
            nval = body.i32()
            body.read(8 * nval)                        # doubles (v8)
            nf = body.i32()
            body.read(12 * nf)                         # (chrIdx int, factor double)
        self.norm_expected_pos = self.master_pos + nbytes + 4
        # normalized expected values live at norm_expected_pos; skip them by
        # re-parsing that section with on-demand reads:
        pos = self.norm_expected_pos
        hdr = Cursor(self.r.read(pos, 1 << 27))
        n_norm_exp = hdr.i32()
        for _ in range(n_norm_exp):
            hdr.cstr(); hdr.cstr(); hdr.i32()          # type, unit, binSize
            nval = hdr.i32()
            hdr.read(8 * nval)
            nf = hdr.i32()
            hdr.read(12 * nf)
        n_norm_vec = hdr.i32()
        self.norm_index = {}                           # (type, chrIdx, unit, res) -> (pos, size)
        for _ in range(n_norm_vec):
            typ = hdr.cstr(); cidx = hdr.i32(); unit = hdr.cstr(); res = hdr.i32()
            p = hdr.i64(); sz = hdr.i32()
            self.norm_index[(typ, cidx, unit, res)] = (p, sz)

    def norm_vector(self, chr_name, norm='KR', res=10000):
        cidx = self.chr_index[chr_name]
        key = (norm, cidx, 'BP', res)
        if key not in self.norm_index:
            return None
        pos, size = self.norm_index[key]
        cur = Cursor(self.r.read(pos, size))
        n = cur.i32()
        return np.array([cur.f64() for _ in range(n)], dtype=np.float64)

    def _zoom_index(self, chr_name, res):
        """Return (blockBinCount, blockColumnCount, {blockNumber: (pos, size)})."""
        cidx = self.chr_index[chr_name]
        key = f'{cidx}_{cidx}'
        mpos, msize = self.master[key]
        cur = Cursor(self.r.read(mpos, 12))
        c1 = cur.i32(); c2 = cur.i32(); nres = cur.i32()
        fp = mpos + 12
        for _ in range(nres):
            hdr_bytes = self.r.read(fp, 1 << 16)
            hc = Cursor(hdr_bytes)
            unit = hc.cstr(); hc.i32()
            hc.f32(); hc.f32(); hc.f32(); hc.f32()
            binsize = hc.i32(); bbc = hc.i32(); bcc = hc.i32(); nb = hc.i32()
            hdr_len = hc.o + nb * 16
            if len(hdr_bytes) < hdr_len:
                hc = Cursor(self.r.read(fp, hdr_len))
                hc.o = hdr_len - nb * 16
            blocks = {}
            for _ in range(nb):
                bn = hc.i32(); bp = hc.i64(); bs = hc.i32()
                blocks[bn] = (bp, bs)
            if unit == 'BP' and binsize == res:
                return bbc, bcc, blocks
            fp += hdr_len
        raise KeyError(f'resolution {res} not found for {chr_name}')

    def _parse_block(self, buf):
        raw = zlib.decompress(buf)
        cur = Cursor(raw)
        nrec = cur.i32()
        recs = []
        if self.version < 7:
            for _ in range(nrec):
                recs.append((cur.i32(), cur.i32(), cur.f32()))
            return recs
        ox = cur.i32(); oy = cur.i32()
        use_short = cur.u8() == 0
        typ = cur.u8()
        if typ == 2:
            npts = cur.i32(); w = cur.i16()
            for i in range(npts):
                row, col = divmod(i, w)
                if use_short:
                    c = cur.i16()
                    if c == -32768: continue
                    recs.append((ox + col, oy + row, float(c)))
                else:
                    c = cur.f32()
                    if c != c: continue
                    recs.append((ox + col, oy + row, c))
            return recs
        # type 1: row-major with short/int deltas; v8 -> binX/binY both int-or-short
        # determined how? v8 lacks the v9 useShortBinX/Y bytes; in practice v8 files
        # use both-short encoding when values fit. BinReader v<9 path uses
        # useShortBinX=useShortBinY=true defaults from readBlock.
        rowcount = cur.i16()
        for _ in range(rowcount):
            by = oy + cur.i16()
            colcount = cur.i16()
            for _ in range(colcount):
                bx = ox + cur.i16()
                c = cur.i16() if use_short else cur.f32()
                recs.append((bx, by, float(c)))
        return recs

    def matrix(self, chr_name, res=10000, norm='KR'):
        """Dense KR-normalized intra-chromosomal matrix (upper triangle mirrored)."""
        cidx = self.chr_index[chr_name]
        clen = self.chroms[cidx][1]
        nbins = int(np.ceil(clen / res))
        bbc, bcc, blocks = self._zoom_index(chr_name, res)
        nv = self.norm_vector(chr_name, norm, res) if norm else None
        M = np.zeros((nbins, nbins), dtype=np.float32)
        # fetch near-diagonal blocks: |blockRow - blockCol| <= 1 covers +-1000 bins
        want = {}
        for bn, (bp, bs) in blocks.items():
            brow, bcol = divmod(bn, bcc)
            if abs(brow - bcol) <= 1:
                want[bn] = (bp, bs)
        # span-fetch the needed byte range in chunks
        if want:
            lo = min(p for p, _ in want.values()); hi = max(p + s for p, s in want.values())
            span = bytearray()
            CH = 1 << 24
            p = lo
            while p < hi:
                n = min(CH, hi - p)
                span += self.r.read(p, n)
                p += n
            for bn, (bp, bs) in want.items():
                buf = bytes(span[bp - lo: bp - lo + bs])
                for bx, by, c in self._parse_block(buf):
                    if bx < nbins and by < nbins:
                        M[by, bx] = c
                        M[bx, by] = c
        if nv is not None:
            nv = nv[:nbins]
            with np.errstate(invalid='ignore', divide='ignore'):
                scale = np.where(np.isfinite(nv) & (nv != 0), nv, np.nan).astype(np.float32)
            # in-place float32 division: a float64 promotion here OOMs on chr1
            M /= scale[:, None]
            M /= scale[None, :]
            M[~np.isfinite(M)] = 0.0
        return M
