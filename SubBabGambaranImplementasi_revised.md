# 3.6 Gambaran Implementasi

Penelitian ini bersifat analisis, namun untuk memberikan konteks penerapan di lapangan, bagian ini menjelaskan gambaran konseptual bagaimana model Intrusion Detection System (IDS) dari hasil penelitian dapat diimplementasikan pada Wireless Sensor Networks (WSN) di lapangan. Penjelasan ini berfokus pada bagaimana fitur-fitur yang digunakan pada dataset WSN-DS diperoleh dari proses komunikasi aktual antar node, serta bagaimana model machine learning dapat dijalankan secara efisien pada sistem dengan keterbatasan sumber daya.

## 3.6.1 Justifikasi Deployment Model pada Sensor Node

Berdasarkan evaluasi latensi inferensi yang dilakukan pada mikrokontroler MSP430F5529, penelitian ini membuktikan bahwa model machine learning berbasis ensemble tree dapat di-deploy langsung pada setiap sensor node. Tabel 3.X menunjukkan hasil evaluasi kelayakan deployment model pada mikrokontroler.

| Model | Waktu Inferensi | Penggunaan Flash | Penggunaan RAM |
|-------|-----------------|------------------|----------------|
| Extra Trees | 50,8 µs | 8,31% (10.893 bytes) | 2,73% (224 bytes) |
| Random Forest | 50,8 µs | 8,31% (10.895 bytes) | 2,73% (224 bytes) |
| Gradient Boosting | 121,6 µs | 15,45% (20.247 bytes) | 2,73% (224 bytes) |
| Histogram Gradient Boosting | 121,6 µs | 6,36% (8.335 bytes) | 2,73% (224 bytes) |

Hasil evaluasi menunjukkan bahwa waktu inferensi model Extra Trees dan Random Forest hanya membutuhkan 50,8 µs, jauh lebih cepat dibandingkan dengan interval waktu minimum antar paket pada protokol LEACH yang berkisar 100 ms. Hal ini memberikan margin keamanan sebesar 2.000 kali lipat, memastikan bahwa proses deteksi intrusi tidak mengganggu operasi normal komunikasi jaringan.

Pendekatan deployment model pada setiap sensor node dipilih karena karakteristik protokol LEACH dimana setiap node memiliki potensi untuk terpilih menjadi Cluster Head (CH) di setiap ronde. Oleh karena itu, setiap node harus memiliki kapabilitas IDS yang sama agar dapat menjalankan fungsi deteksi intrusi ketika terpilih menjadi CH.

## 3.6.2 Peran Node dalam Sistem IDS

Dalam arsitektur LEACH, terdapat pembagian peran yang jelas antara node biasa (member node) dan Cluster Head (CH). Pembagian peran ini mempengaruhi kapan dan bagaimana inferensi IDS dilakukan.

### Node Biasa (Member Node)

Ketika sebuah node tidak terpilih menjadi CH pada suatu ronde, node tersebut berperan sebagai member node dengan tugas:

1. **Pengumpulan Data Sensor**: Mengumpulkan data dari sensor (suhu, kelembaban, dll.) sesuai fungsi utama WSN.

2. **Pengumpulan Fitur Komunikasi Lokal**: Mencatat aktivitas komunikasi yang relevan dengan dirinya sendiri, meliputi:
   - `Advertise_Receive`: Jumlah advertisement yang diterima dari CH
   - `Join_Request_Sent`: Jumlah request bergabung yang dikirim
   - `Schedule_Receive`: Jadwal TDMA yang diterima dari CH
   - `DATA_Sent`: Jumlah paket data yang dikirim ke CH
   - `Rank`: Urutan slot TDMA yang diterima
   - `Remaining_Energy`: Estimasi sisa energi node sendiri

3. **Passive Monitoring**: Melalui mekanisme *passive overhearing* pada protokol CSMA-MAC, node dapat mendeteksi aktivitas node tetangga dalam jangkauan radio, seperti jumlah transmisi yang terdeteksi.

4. **Pelaporan ke CH**: Mengirimkan data sensor beserta informasi status komunikasi ke CH sesuai jadwal TDMA.

**Pada fase ini, node biasa TIDAK melakukan inferensi IDS**, melainkan hanya mengumpulkan dan mengirimkan data.

### Cluster Head (CH)

Ketika sebuah node terpilih menjadi CH pada suatu ronde, node tersebut memiliki tanggung jawab tambahan dan **menjalankan inferensi IDS**. CH memiliki "full view" terhadap aktivitas seluruh cluster karena:

1. **Menerima Data dari Semua Member**: CH menerima laporan dari setiap node dalam cluster, sehingga dapat melihat pola komunikasi secara agregat.

2. **Mengumpulkan Fitur Khusus CH**: Beberapa fitur hanya dapat diperoleh oleh CH:
   - `Advertise_Sent`: Jumlah advertisement yang di-broadcast
   - `Join_Request_Receive`: Jumlah request bergabung yang diterima
   - `Schedule_Sent`: Jadwal TDMA yang di-broadcast
   - `DATA_Receive`: Jumlah paket data yang diterima dari member
   - `Data_Sent_To_BS`: Jumlah paket yang dikirim ke Base Station
   - `distance_CH_To_BS`: Jarak CH ke Base Station

3. **Menjalankan Inferensi IDS**: Dengan informasi lengkap dari seluruh cluster, CH dapat menjalankan model IDS untuk mendeteksi:
   - **Flooding Attack**: Lonjakan abnormal pada `Join_Request_Receive`
   - **TDMA Attack**: Transmisi di luar jadwal yang terdeteksi
   - **Blackhole Attack**: Node yang menerima tapi tidak meneruskan data
   - **Grayhole Attack**: Pola selektif dalam dropping paket

4. **Mengambil Tindakan**: Berdasarkan hasil deteksi, CH dapat:
   - Mengabaikan paket dari node yang terdeteksi sebagai penyerang
   - Mengirim alert ke Base Station
   - Mengecualikan node mencurigakan dari jadwal TDMA ronde berikutnya

## 3.6.3 Arsitektur Komunikasi WSN

Dalam sistem WSN berbasis LEACH, komunikasi diatur dalam dua fase utama:

### Fase Setup (Pembentukan Cluster)

Pada fase ini, komunikasi menggunakan protokol CSMA-MAC:

1. **Advertisement Phase**: Node yang terpilih menjadi CH mem-broadcast pesan advertisement.
2. **Cluster Join Phase**: Node biasa memilih CH berdasarkan kekuatan sinyal (RSSI) dan mengirim `Join_Request`.
3. **Schedule Creation**: CH membuat jadwal TDMA dan mem-broadcast ke semua member.

### Fase Steady-State (Transmisi Data)

Pada fase ini, komunikasi menggunakan protokol TDMA:

1. Node mengirim data ke CH sesuai slot waktu yang dijadwalkan.
2. CH mengagregasi data dan mengirim ke Base Station.
3. Proses berlanjut hingga ronde berikutnya dimulai.

Beberapa fitur seperti `Advertise_Sent/Receive`, `Join_Request_Sent/Receive`, `Schedule_Sent/Receive` didapatkan pada MAC layer selama fase setup. Sedangkan fitur seperti `DATA_Sent/Receive`, `Rank` didapatkan selama fase steady-state.

## 3.6.4 Arsitektur Sistem IDS pada LEACH

Secara diagram, arsitektur sistem IDS dapat digambarkan pada Gambar 3.7. Arsitektur tersebut dapat dijelaskan sebagai berikut:

### 1. Node Sensor dengan Embedded IDS

Setiap node sensor berbasis mikrokontroler MSP430F5529 (atau setara) dilengkapi dengan:

- **Firmware Dual-Mode**: Kode program yang dapat beroperasi dalam dua mode:
  - **Mode Member**: Fokus pada pengumpulan data dan pelaporan ke CH
  - **Mode CH**: Menjalankan fungsi agregasi dan inferensi IDS

- **Model IDS Tertanam**: Model ensemble tree yang telah dikuantisasi disimpan dalam Flash memory dan siap digunakan ketika node terpilih menjadi CH.

- **Buffer Fitur**: Menyimpan fitur-fitur komunikasi yang dikumpulkan selama operasi.

### 2. Alur Kerja pada Mode Member

```
┌─────────────────────────────────────────────────────────────────┐
│                    NODE DALAM MODE MEMBER                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │ Terima Adv    │───▶│ Kirim Join    │───▶│ Terima        │   │
│  │ dari CH       │    │ Request       │    │ Schedule TDMA │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Catat Fitur Lokal ke Buffer                │   │
│  │  (Adv_Recv, Join_Sent, Sched_Recv, Rank, Energy)       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │     Fase Steady-State: Kirim Data sesuai Slot TDMA      │   │
│  │     (Data sensor + Status komunikasi ke CH)             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                 │
│  * TIDAK ada inferensi IDS pada mode ini                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Alur Kerja pada Mode Cluster Head

```
┌─────────────────────────────────────────────────────────────────┐
│                    NODE DALAM MODE CLUSTER HEAD                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FASE SETUP:                                                    │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │ Broadcast     │───▶│ Terima Join   │───▶│ Broadcast     │   │
│  │ Advertisement │    │ Request       │    │ Schedule TDMA │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │    Catat Fitur CH: Adv_Sent, Join_Recv, Sched_Sent     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                 │
│  FASE STEADY-STATE:                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           Terima Data dari Member Node                   │   │
│  │        (Catat: DATA_Recv per node, pola timing)         │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              INFERENSI IDS (50,8 µs)                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │ Kuantisasi  │─▶│  Traversal  │─▶│   Voting    │      │   │
│  │  │ 18 Fitur    │  │  5 Trees    │  │  Majority   │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│              ┌────────────┴────────────┐                       │
│              ▼                         ▼                       │
│     ┌──────────────┐          ┌──────────────┐                 │
│     │   Normal     │          │   Serangan   │                 │
│     │   (Kelas 0)  │          │  (Kelas 1-4) │                 │
│     └──────┬───────┘          └──────┬───────┘                 │
│            │                         │                         │
│            ▼                         ▼                         │
│     ┌──────────────┐          ┌──────────────┐                 │
│     │ Agregasi &   │          │ Drop paket,  │                 │
│     │ Kirim ke BS  │          │ Alert ke BS  │                 │
│     └──────────────┘          └──────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Base Station / Sink Node

Base station berfungsi sebagai pengumpul data agregasi dari semua CH dan penghubung ke jaringan eksternal:

- Menerima data dan laporan deteksi serangan dari setiap CH
- Menyimpan log untuk analisis historis
- Meneruskan informasi ke Control Center jika diperlukan

### 5. Control Center

Sistem backend yang berfungsi untuk:
- Menyimpan log aktivitas jaringan dan laporan deteksi serangan
- Melakukan analisis tren serangan secara historis
- Mengelola pembaruan model IDS jika diperlukan
- Menyediakan dashboard monitoring untuk administrator

## 3.6.5 Mekanisme Pengumpulan Fitur

Tabel 3.X menjelaskan metode pengambilan untuk setiap fitur yang digunakan oleh model IDS, beserta keterangan node mana yang dapat mengumpulkan fitur tersebut:

| Fitur | Sumber | Metode Pengambilan |
|-------|--------|-------------------|
| `Id` | Semua | Ditentukan saat inisialisasi jaringan (alamat MAC atau ID unik) |
| `Time` | Semua | Timestamp dari local clock yang disinkronkan via beacon dari CH |
| `Is_CH` | Semua | Ditentukan oleh algoritma probabilistik LEACH di awal setiap ronde |
| `Who_CH` | Member | ID CH yang dipilih, diperoleh dari pesan advertisement yang diterima |
| `Distance_To_CH` | Member | Estimasi jarak berdasarkan RSSI dari pesan advertisement CH |
| `Advertise_Sent` | **CH only** | Counter internal CH saat broadcast advertisement |
| `Advertise_Receive` | Member | Counter saat menerima advertisement (biasanya 1 per ronde) |
| `Join_Request_Sent` | Member | Counter saat mengirim request bergabung ke CH |
| `Join_Request_Receive` | **CH only** | Counter saat menerima request dari member |
| `Schedule_Sent` | **CH only** | Counter saat broadcast jadwal TDMA |
| `Schedule_Receive` | Member | Counter saat menerima jadwal TDMA |
| `Rank` | Member | Nomor slot TDMA yang ditetapkan CH, diekstrak dari Schedule |
| `DATA_Sent` | Member | Counter paket data yang dikirim ke CH |
| `DATA_Receive` | **CH only** | Counter paket data yang diterima dari semua member |
| `Data_Sent_To_BS` | **CH only** | Counter paket agregasi yang dikirim ke Base Station |
| `distance_CH_To_BS` | **CH only** | Estimasi jarak ke BS berdasarkan RSSI atau informasi topologi |
| `send_code` | Semua | Kode autentikasi yang ditetapkan saat fase advertisement |
| `Expanded_Energy` | Semua* | Lihat penjelasan di bawah |

### Catatan tentang Fitur `Expanded_Energy`

Fitur `Expanded_Energy` (energi yang telah dikonsumsi) memerlukan penjelasan khusus:

1. **Untuk Node Sendiri**: Setiap node dapat mengestimasi konsumsi energinya sendiri berdasarkan:
   - Jumlah transmisi yang dilakukan (TX cost)
   - Jumlah penerimaan (RX cost)
   - Waktu aktif dan sleep mode
   - Model energi yang telah dikalibrasi untuk hardware yang digunakan

2. **Untuk Node Lain**: CH memperoleh informasi energi node member melalui dua mekanisme:
   - **Self-reporting**: Member menyertakan informasi sisa energi dalam paket data yang dikirim ke CH
   - **Estimasi**: CH dapat mengestimasi berdasarkan jumlah transmisi yang diterima dari node tersebut

### Konstruksi Vektor Fitur untuk Inferensi

Ketika CH akan menjalankan inferensi IDS untuk mengevaluasi suatu node member, CH mengkonstruksi vektor 18 fitur dengan menggabungkan:

1. **Data yang dilaporkan node** (via paket data ke CH):
   - Aktivitas komunikasi node tersebut
   - Estimasi energi node

2. **Observasi CH terhadap node tersebut**:
   - Jumlah `DATA_Receive` dari node tersebut
   - Timing transmisi (apakah sesuai slot TDMA)
   - Jumlah `Join_Request` dari node tersebut

3. **Konteks cluster**:
   - Informasi CH (`Who_CH`, `Distance_To_CH` dari perspektif node)
   - `Rank` yang diberikan kepada node tersebut

## 3.6.6 Proses Konversi Model untuk Deployment

Untuk dapat dijalankan pada mikrokontroler dengan sumber daya terbatas, model machine learning hasil pelatihan perlu melalui proses konversi sebagai berikut:

### Tahap 1: Kuantisasi INT8

Model yang dilatih menggunakan floating-point 64-bit dikonversi ke integer 8-bit menggunakan teknik min-max quantization:

```
q_value = round((float_value - min_value) × scale) + zero_point
```

dimana:
- `scale = 255 / (max_value - min_value)`
- `zero_point = -min_value × scale`

Proses ini mengurangi ukuran model hingga 8 kali lipat tanpa degradasi akurasi yang signifikan untuk model berbasis pohon keputusan.

### Tahap 2: Pembatasan Kedalaman Pohon

Untuk memastikan model dapat dimuat dalam memori Flash mikrokontroler, kedalaman pohon dibatasi maksimal 6 level. Dengan kedalaman ini, setiap pohon memiliki maksimal 63 node (2^6 - 1), dengan setiap node membutuhkan 6 bytes penyimpanan.

### Tahap 3: Generasi Kode C

Model yang telah dikuantisasi dikonversi menjadi kode C yang dapat dikompilasi untuk arsitektur MSP430. Struktur data pohon disimpan sebagai array byte dengan format:

```c
// Format node: [feature_idx, threshold, left_lo, left_hi, right_lo, right_hi]
static const uint8_t tree_0[] = {
    1, 127, 1, 0, 2, 0,    // Node 0: if feature[1] <= 127, go left, else right
    5, 64, 3, 0, 4, 0,     // Node 1: if feature[5] <= 64, go left, else right
    255, 0, 0, 0, 0, 0,    // Node 2: LEAF - predict class 0 (Normal)
    255, 1, 0, 0, 0, 0,    // Node 3: LEAF - predict class 1 (Flooding)
    ...
};
```

### Tahap 4: Kompilasi dan Flashing

Kode C dikompilasi menggunakan toolchain `msp430-elf-gcc` dengan optimisasi `-O2` dan di-flash ke memori mikrokontroler. Model disimpan di Flash memory dan diakses saat node beroperasi sebagai CH.

## 3.6.7 Skenario Deteksi Serangan

Berikut adalah contoh skenario bagaimana sistem IDS mendeteksi berbagai jenis serangan:

### Deteksi Flooding Attack

```
Skenario:
- Node penyerang mengirim banyak Join_Request untuk membanjiri CH

Proses Deteksi:
1. CH mencatat: Join_Request_Receive = 50 (abnormal, biasanya 1 per node)
2. CH mengkonstruksi fitur untuk node tersebut
3. Inferensi model: fitur Join_Request_Receive yang tinggi → Kelas 1 (Flooding)
4. Respons: CH mengabaikan Join_Request dari node tersebut
```

### Deteksi TDMA Attack

```
Skenario:
- Node penyerang mengirim data di luar slot TDMA yang dijadwalkan

Proses Deteksi:
1. CH mendeteksi transmisi di luar slot yang dijadwalkan untuk node tersebut
2. Fitur timing dan DATA_Receive mencerminkan anomali
3. Inferensi model: pola transmisi tidak sesuai jadwal → Kelas 4 (TDMA Attack)
4. Respons: CH tidak memproses data dari transmisi ilegal
```

### Deteksi Blackhole Attack

```
Skenario:
- Node yang seharusnya meneruskan data (jika multi-hop) tidak meneruskan

Proses Deteksi:
1. CH melihat node menerima banyak data (dari passive monitoring) tapi 
   tidak meneruskan
2. Rasio DATA_Receive vs DATA_Sent tidak wajar
3. Inferensi model: pola blackhole → Kelas 2 (Blackhole Attack)
4. Respons: Rute data dialihkan dari node tersebut
```

## 3.6.8 Perbandingan: IDS di CH vs IDS di Gateway

| Aspek | IDS di Cluster Head | IDS di Gateway |
|-------|---------------------|----------------|
| **Latensi Deteksi** | ~50 µs (per cluster) | >100 ms (setelah data sampai gateway) |
| **Cakupan Deteksi** | Per-cluster, terdistribusi | Terpusat, seluruh jaringan |
| **Kemampuan Respons** | Langsung (drop paket, exclude node) | Terlambat (serangan sudah terjadi) |
| **Beban Komputasi** | Terdistribusi ke setiap CH | Terpusat di gateway |
| **Ketahanan** | Fault-tolerant (banyak CH) | Single point of failure |
| **Kompatibilitas LEACH** | Sesuai (CH berganti setiap ronde) | Tidak relevan dengan rotasi CH |
| **Konsumsi Energi Node** | Minimal (inferensi hanya saat jadi CH) | Tidak ada tambahan di node |
| **Kompleksitas Deployment** | Setiap node perlu model | Hanya gateway perlu model |

Pendekatan IDS di CH lebih sesuai untuk arsitektur LEACH karena:
1. Memanfaatkan posisi strategis CH yang memiliki "full view" cluster
2. Memungkinkan respons langsung terhadap serangan
3. Setiap node sudah harus siap menjadi CH, sehingga memiliki model adalah keharusan

## 3.6.9 Kesimpulan

Melalui mekanisme yang telah dijelaskan, sistem IDS pada arsitektur LEACH beroperasi dengan prinsip berikut:

1. **Semua node memiliki model IDS** yang tertanam dalam firmware, karena setiap node berpotensi menjadi CH.

2. **Inferensi dilakukan hanya oleh CH**, karena CH memiliki informasi lengkap tentang aktivitas seluruh cluster dan berada pada posisi strategis untuk mendeteksi anomali.

3. **Node member fokus pada pengumpulan data** dan melaporkan status komunikasinya ke CH, tanpa overhead inferensi.

4. **Waktu inferensi 50,8 µs** memberikan margin yang sangat besar terhadap timing constraint LEACH, memastikan fungsi IDS tidak mengganggu operasi normal CH.

5. **Penggunaan memori < 9% Flash** memastikan masih tersedia ruang yang cukup untuk firmware komunikasi, sensor driver, dan fungsi lainnya.

Pendekatan ini memastikan keseimbangan antara keamanan jaringan, efisiensi energi, dan kesesuaian dengan karakteristik protokol LEACH dimana peran CH bersifat rotasional.
