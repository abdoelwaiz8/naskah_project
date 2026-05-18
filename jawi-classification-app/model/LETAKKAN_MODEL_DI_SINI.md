# Placeholder — Letakkan file resnet34_jawi.pth di sini

Folder ini adalah tempat menyimpan file bobot (weights) model PyTorch Anda.

## Cara mendapatkan file bobot:
1. Latih model ResNet34 sendiri menggunakan dataset Jawi Anda.
2. Simpan bobot menggunakan: `torch.save(model.state_dict(), "resnet34_jawi.pth")`
3. Letakkan file `.pth` hasil training di dalam folder ini.

## Struktur yang diharapkan:
```
model/
└── resnet34_jawi.pth   ← File bobot model Anda
```

## Catatan:
- File `.pth` berukuran sekitar 80-90 MB untuk ResNet34.
- Pastikan jumlah kelas output sesuai dengan `NUM_CLASSES` di `app/inference.py`.
