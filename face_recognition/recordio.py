"""Minimal MXNet RecordIO reader without requiring MXNet."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import struct

from PIL import Image
from PIL import UnidentifiedImageError
from torch.utils.data import Dataset

HEADER = struct.Struct("<IfQQ")

@dataclass(frozen=True)
class IRHeader:
    flag: int; label: float; id: int; id2: int

def unpack_header(raw):
    return IRHeader(*HEADER.unpack(raw[:HEADER.size]))

class RecordIODataset(Dataset):
    def __init__(self, rec_path, idx_path, transform=None, max_samples=None):
        self.rec_path, self.transform = Path(rec_path), transform
        raw = Path(idx_path).read_bytes()
        # MXIndexedRecordIO writes its index as tab-separated key/offset lines.
        # Retain a binary fallback for older third-party RecordIO exporters.
        try:
            lines = raw.decode("ascii").splitlines()
            self.offsets = [int(line.split("\t", 1)[1]) for line in lines if "\t" in line]
        except (UnicodeDecodeError, ValueError):
            self.offsets = [struct.unpack_from("<Q", raw, offset + 8)[0] for offset in range(0, len(raw) - 15, 16)]
        self.offsets = self._image_offsets(self.offsets, max_samples)
        if not self.offsets: raise ValueError("RecordIO index contains no records")

    def _read_header(self, handle, offset):
        handle.seek(offset)
        # Each MXNet RecordIO entry begins with a 4-byte record marker,
        # followed by its length and then the 24-byte IRHeader.
        handle.read(4)
        length = struct.unpack("<I", handle.read(4))[0] & 0x00FFFFFF
        return unpack_header(handle.read(length)[:HEADER.size])

    def _image_offsets(self, offsets, max_samples):
        image_offsets = []
        with self.rec_path.open("rb") as handle:
            for offset in offsets:
                if self._read_header(handle, offset).flag == 0:
                    image_offsets.append(offset)
                    if max_samples is not None and len(image_offsets) >= max_samples:
                        break
        return image_offsets
    def __len__(self): return len(self.offsets)
    def labels(self):
        """Return identity labels without decoding every training image."""
        values = []
        with self.rec_path.open("rb") as handle:
            for offset in self.offsets:
                label = self._read_header(handle, offset).label
                if int(label) != label:
                    raise ValueError("RecordIO labels must be integral identity IDs")
                values.append(int(label))
        return values
    def __getitem__(self, index):
        with self.rec_path.open("rb") as handle:
            handle.seek(self.offsets[index])
            handle.read(4)
            length = struct.unpack("<I", handle.read(4))[0] & 0x00FFFFFF
            data = handle.read(length)
        header = unpack_header(data); label = int(header.label)
        if label != header.label: raise ValueError("RecordIO labels must be integral identity IDs")
        encoded = data[HEADER.size:]
        # Some RecordIO exporters retain alignment bytes before JPEG data.
        jpeg_start = encoded.find(b"\xff\xd8", 0, 64)
        if jpeg_start >= 0:
            encoded = encoded[jpeg_start:]
        try:
            image = Image.open(BytesIO(encoded)).convert("RGB")
        except UnidentifiedImageError as error:
            raise ValueError(
                f"RecordIO entry {index} at offset {self.offsets[index]} has no decodable image; "
                f"flag={header.flag}, prefix={encoded[:16].hex()}"
            ) from error
        return self.transform(image) if self.transform else image, label
