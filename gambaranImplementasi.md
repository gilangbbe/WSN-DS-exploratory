3.6 Gambaran Implementasi
Penelitian ini bersifat analisis, namun untuk memberikan konteks penerapan di lapangan, bagian ini menjelaskan gambaran konseptual bagaimana model Intrusion Detection System (IDS) dari hasil penelitian dapat diimplementasikan pada Wireless Sensor Networks (WSN). Penjelasan difokuskan pada bagaimana fitur-fitur yang digunakan pada dataset WSN-DS diperoleh dari proses komunikasi aktual antar node, serta bagaimana model machine learning dapat dijalankan secara efisien pada sistem dengan keterbatasan sumber daya.
Dalam sistem WSN, setiap node sensor berperan memantau kondisi lingkungan dan mengirimkan data ke Cluster Head (CH). Komunikasi antar node diatur menggunakan protokol Carrier Sense Multiple Access – Medium Access Control (CSMA-MAC) yang memastikan node hanya mengirim ketika kanal tidak sibuk, sehingga mengurangi collision dan menjaga efisiensi energi. Setelah proses pembentukan cluster, pengiriman data dilakukan berdasarkan jadwal Time Division Multiple Access (TDMA) yang dikendalikan oleh CH.
Beberapa fitur seperti Advertise_Sent/Receive, Join_Request, dan Schedule_Sent/Receive diperoleh dari MAC layer. Sementara itu, network layer yang menangani routing antar node mengatur fitur-fitur seperti DATA_Sent/Receive, Distance_to_CH, Send_Code, dan Rank. Model IDS hasil penelitian menggunakan fitur-fitur tersebut untuk mendeteksi pola komunikasi abnormal, seperti paket yang dikirim di luar jadwal TDMA atau lonjakan JOIN_Request yang mengindikasikan adanya serangan.
Secara konseptual, sistem IDS dirancang sesuai dengan karakteristik protokol LEACH, di mana CH dipilih secara acak dan bergantian setiap ronde untuk menyeimbangkan konsumsi energi. Oleh karena itu, model IDS tidak diimplementasikan langsung pada CH, melainkan dijalankan pada sink node atau gateway yang memiliki kapasitas komputasi lebih tinggi dan tidak berganti peran selama operasi jaringan.
Secara diagram, arsitektur implementasi ditunjukkan pada Gambar 3.7.

Arsitektur Implementasi IDS
Arsitektur sistem dapat dijelaskan sebagai berikut:
1. Node Sensor
Node sensor berkomunikasi menggunakan protokol CSMA-MAC pada fase pembentukan cluster dan TDMA pada fase transmisi data. Node mengumpulkan informasi komunikasi seperti:
Advertise_Sent
Join_Request_Sent
Data_Sent
Estimasi Expanded Energy
Data mentah ini dikirim ke CH yang terpilih pada ronde tersebut.
2. Cluster Head (CH)
CH berfungsi sebagai pengumpul data dari node-node dalam satu cluster. Karena CH berganti setiap ronde sesuai mekanisme LEACH, perannya terbatas pada agregasi sementara data dan meneruskannya ke sink node atau base station.
3. Base Station / Sink Node
Base station terhubung langsung ke LoRa Gateway dan memiliki dua modul komunikasi:
Short-range radio interface (misalnya IEEE 802.15.4, ZigBee, atau nRF24L01) untuk menerima data agregasi dari CH
Long-range radio interface untuk meneruskan data ke gateway
Perangkat keras yang umum digunakan berupa single-board computer seperti Raspberry Pi atau Jetson Nano, yang memiliki kapasitas komputasi cukup untuk pra-pemrosesan dan penyimpanan data sementara.
4. LoRa Gateway
LoRa Gateway merupakan lokasi utama dijalankannya model IDS. Gateway menerima data hasil agregasi dari base station melalui kanal LoRa, kemudian menjalankan model machine learning hasil penelitian untuk mendeteksi serangan. Model diekspor dalam format ringan seperti joblib, ONNX, atau TensorFlow Lite agar dapat di-embed secara efisien. Hasil deteksi diteruskan ke control center.
5. Control Center
Merupakan sistem backend yang menyimpan catatan aktivitas jaringan serta laporan hasil deteksi serangan.

Mekanisme Monitoring dan Pengambilan Data
Untuk mendapatkan data pelatihan model IDS, diperlukan layanan monitoring dengan biaya serendah mungkin namun tetap mampu menghasilkan data yang representatif. Dalam arsitektur ini, setiap node sensor tidak hanya bertindak sebagai pengirim data, tetapi juga berpartisipasi dalam proses monitoring.
Setiap node memantau sekumpulan node di sekitarnya untuk mencatat aktivitas komunikasi, seperti jumlah paket yang dikirim, diterima, atau hilang selama transmisi. Pendekatan ini bertujuan mendistribusikan beban monitoring agar tidak menimbulkan konsumsi energi berlebih pada node tertentu.
Seluruh komunikasi dikendalikan oleh protokol CSMA-MAC, yang memungkinkan node mendeteksi kondisi kanal sebelum mengirim paket. Meskipun hanya satu node yang dapat mentransmisikan dalam satu waktu, node lain tetap berada dalam mode listening dan dapat melakukan passive overhearing. Informasi ini kemudian dikumpulkan oleh CH pada setiap ronde, sehingga terbentuk dataset terdistribusi yang mencerminkan perilaku komunikasi jaringan secara menyeluruh.

Tabel 3.4 Metode Pengambilan Fitur
| Fitur                                    | Metode Pengambilan Fitur                                                    |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| Id                                       | Ditentukan saat inisialisasi jaringan atau berdasarkan alamat MAC           |
| Time                                     | Waktu aktivitas komunikasi, dicatat melalui *local clock* yang disinkronkan |
| Is_CH                                    | Ditentukan oleh algoritma LEACH                                             |
| Who_CH                                   | Diperoleh saat node menerima pesan advertisement dari CH                    |
| Distance_To_CH                           | Estimasi jarak ke CH berdasarkan RSSI                                       |
| Advertise_Sent / Advertise_Receive       | Diperoleh dari buffer komunikasi MAC                                        |
| JOIN_Request_Sent / JOIN_Request_Receive | Diperoleh dari buffer komunikasi MAC                                        |
| Schedule_Sent / Schedule_Receive         | Diperoleh dari buffer komunikasi MAC                                        |
| Rank                                     | Ditentukan oleh protokol TDMA                                               |
| DATA_Sent / DATA_Receive                 | Dicatat dari aktivitas sendiri dan node tetangga                            |
| Data_Sent_To_BS                          | Dihitung dari total paket uplink                                            |
| distance_CH_To_BS                        | Estimasi jarak CH ke Base Station                                           |
| send_code                                | Ditetapkan saat fase advertisement                                          |
| Expanded Energy                          | Diperoleh dari sensor konsumsi energi node                                  |


Melalui mekanisme ini, seluruh node sensor berkontribusi dalam pengumpulan data tanpa menimbulkan overhead signifikan. Data hasil pengamatan dikirim ke CH setiap ronde, diteruskan ke base station dan gateway, lalu digunakan untuk inferensi oleh model IDS.