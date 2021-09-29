# train
### train.csv
* BYJ, HYN, SJW이 각각 자신의 이름을 10번 외치는 데이터

# test
### train_BYJ.csv
* BYJ이 SJW 이름을 10번 외치는 데이터  
* BYJ이 HYN 이름을 10번 외치는 데이터
### train_HYN.csv
* HYN이 SJW 이름을 10번 외치는 데이터  
* HYN이 BYJ 이름을 10번 외치는 데이터
### train_SJW.csv
* SJW이 BYJ 이름을 10번 외치는 데이터  
* SJW이 HYN 이름을 10번 외치는 데이터  

# 사용한 agent
* Python Praat - parselmouth Package

# feature
### Label	
* 실제 외친 사람
### Formant Frequency - F0(pitch), F1, F2, F3, F4, F5
* 성대가 울릴 때 성도가 공진되면서 특히나 강한 크기를 보이는 일련의 주파수들이 있는데, 낮은 주파수들부터 F0, F1, F2, F3, F4, F5라고 한다.  
* 이러한 주파수들은 말하는 사람과 단어 등에 따라 달라져서, 이를 구분하는 데에 사용할 수 있다!
