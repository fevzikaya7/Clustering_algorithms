import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel("sample_not.xlsx", sheet_name="Sheet2")
data_frame = pd.DataFrame(data)
st_array = np.matrix(data_frame) 

def merkez_bulma(st_array,k):
    merkez_index = []
    merkez = []
    aralik = (len(st_array)//(k+1))
    first = aralik

    for key in range(0,k):
        merkez_index.append(first-1)
        first = first + aralik

    for key in range(0,k):
        merkez.append(st_array[merkez_index[key]])

    return merkez 

def k_means(st_array):
    iteration = []
    sonuc = []
    wcss = []
    siluet_score = []
    cluster_column = np.zeros((100,1), dtype="int64")
    st_array = np.append(st_array, cluster_column, axis=1)
    for k in range(2,10):  # Her k değeri için algoritma tekrar çalışacak.
        sayac = 0
        st_array_cpy = np.copy(st_array)
        cluster = []
        merkez = merkez_bulma(st_array[:,1:4],k) 
        while True:  # Kümeler oluşana kadar devam edecek.
            sayac = sayac + 1 # Kaçıncı iterasyonda kümelerin oluştuğunu gözlemlemek için
            temp = []  # Noktanın hangi kümeye dahilse o kümenin indexini buraya atacağım.

            for indis in range(0,100):
                mesafe = []  # Noktanın tüm merkezlere olan mesafesini ekleyeceğim. Her nokta için güncellenecek.

                # En yakın merkezi bulup noktayı o kümeye dahil ediyorum.
                # k tane merkez var, en yakını bulup o merkezin indisini küme numarası olarak atıyorum.
                for i in range(0,k):
                    dist = np.linalg.norm(st_array_cpy[indis, 1:4] - merkez[i])
                    mesafe.append(dist)

                # Mesafelerden en yakınını burada buluyorum ve küme index numaralarını içerden listeye ekliyorum.
                temp.append(mesafe.index(min(mesafe))) 

            # Her noktasının kümesini bulduktan sonra bu küme indexlerini verilere ekliyorum.    
            st_array_cpy[:,4] = np.matrix(temp)
            
            if temp == cluster: # Önceki iterasyon sonundaki küme numaraları ile şimdiki küme numaraları aynı ise;
                # Veriler kümelerine ayrılmıştır, k değeri için silüet puanı ve wcss değerini hesaplayalım.
                # Her farklı k değeri için farklı şekilde kümelerine ayrılmış verileri sonuc listesine ekleyelim.

                sonuc.append((k,st_array_cpy))
                wcss.append((k,elbow(st_array_cpy)))
                siluet_score.append((k,silhouette(st_array_cpy,merkez)))
                iteration.append((k,sayac))
                break
            
            cluster = temp 

            # Yeni iterasyona başlamadan merkezleri kümelerin yeni ağırlık merkezlerine taşıyoruz.
            for j in range(0,k):
                temp_df = pd.DataFrame(st_array_cpy)
                temp_mt = np.matrix(temp_df[temp_df[4]==j])
                merkez[j] = np.mean(temp_mt, axis=0)[:,1:4] #Burada küme numarasını temsil eden son kolonu koordinatlara dahil etmiyoruz.
                
    return (sonuc,wcss,siluet_score,iteration) 

def elbow(st_array):
    sum = 0
    for i in range(0,len(st_array)):
        for j in range(0,len(st_array)):
            if st_array[i,4] == st_array[j,4]:
                dist = np.linalg.norm(st_array[i, 1:4] - st_array[j, 1:4])  # Küme numarası içeren kolonu uzaklığa dahil etmiyoruz, dikkat!
                sum += dist**2
    
    return sum 

def silhouette(st_array,merkez):
    s=[]
    for i in range(0,len(st_array)):  # Tüm noktalar sırayla.
        sayac_a=0; sayac_b=0
        a=0; b=0
        temp_merkez = merkez[:]

        # Aynı küme içindeki benzerlik için
        for j in range(0,len(st_array)):
            if (st_array[i,4] == st_array[j,4]):
                sayac_a += 1    # Aynı kümedeki eleman sayısını bulmak için bir sayaç kullandım.
                dist_a = np.linalg.norm(st_array[i, 1:4] - st_array[j, 1:4])
                a += dist_a

        a = a / (sayac_a-1)

        # Kendisine en yakın küme içindeki verilerle benzerliği için
        # Kendi kümesi dışında en yakın merkezi bulmalıyız.
        mesafe = []
        temp_merkez.pop(st_array[i,4])  # İncelediğimiz i noktasının dahil olduğu kümenin merkezini merkez listesinden çıkardım.
        temp_index = st_array[i,4]  # İncelediğimiz i noktasının dahil olduğu kümenin merkezini merkez listesinden çıkardım.
        for m in range(0,len(temp_merkez)):
            uzaklik = np.linalg.norm(st_array[i, 1:4] - temp_merkez[m])
            mesafe.append(uzaklik)  # Diğer merkezlere olan mesafesini ekledim.
            
        if  mesafe.index(min(mesafe)) < temp_index:
            get_index = mesafe.index(min(mesafe))
        else:
            get_index = mesafe.index(min(mesafe)) + 1


        for j in range(0,len(st_array)):
            if (st_array[j,4] == get_index):
                sayac_b += 1  # Bu kümedeki eleman sayısını bulmak için bir sayaç kullandım.
                dist_b = np.linalg.norm(st_array[i, 1:4] - st_array[j, 1:4])
                b += dist_b
        b = b / sayac_b 
        
        s.append((b-a)/max(a,b))

    return (sum(s)/len(st_array))  

k_means(st_array)

# Mat notlarına göre sıralanıp seçilen noktalar eş parçalara böler.
# Mesela k=3 için seçilen noktalar m1=25 , m2=50, m3=75 gibi...

tum_sonuclar = k_means(st_array)
sonuc = tum_sonuclar[0]
wcss = tum_sonuclar[1]
print()
siluet_score = tum_sonuclar[2]
iteration = tum_sonuclar[3] 

clusters = []
wcss_point = []
siluet_point = []
for key in wcss:
    clusters.append(key[0])
    wcss_point.append(key[1])

for key in siluet_score:
    siluet_point.append(key[1])

#############################
plt.plot(clusters, wcss_point)
plt.title('WCSS Point According To k')
plt.xlabel('k')
plt.ylabel('WCSS Point')
plt.show()

plt.plot(clusters, siluet_point)
plt.title('Silhouette Score According To k')
plt.xlabel('k')
plt.ylabel('Silhouette Point')
plt.show() 