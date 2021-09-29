import parselmouth
import pandas as pd

data = pd.DataFrame({
    "times":[],
    "F0(pitch)":[],
    "F1":[],
    "F2":[],
    'F3':[],
    "F4":[],
    "F5":[]
})

formants_value = ['F1','F2',"F3","F4","F5"]

##################################wav파일경로##########################################
soundpath="C:\\future finance\\data\\1.wav"
####################################################################################### 

# 음원파일을 Sound객체로 변환해야 함
Sound = parselmouth.Sound(soundpath)
        
# time_step 변수로 formant 추출 시간단위를 지정함
formant = Sound.to_formant_burg(time_step = 0.1)

# Pitch값을 추출하려면 to_pitch 함수를 사용함
# 추출 방식에 따라 to_pitch_ac, to_pitch_cc 등 여러 함수가 지원됨
# Docs의 API Reference에서 원하는 함수를 찾아 사용하면 됨
pitch = Sound.to_pitch()
        
# formant 추출에 사용된 시간은 ts() 함수로 얻을 수 있음
df = pd.DataFrame({"times":formant.ts()})

# 각 시간대별 F1~F5 얻음
for idx, col in enumerate(formants_value, 1):
    df[col] = df['times'].map(lambda x: formant.get_value_at_time(formant_number = idx, time = x))
        
# F0는 Formant 객체가 아니라 Pitch 객체에서 얻을 수 있음
df['F0(pitch)'] = df['times'].map(lambda x: pitch.get_value_at_time(time = x))
data = data.append(df)

######################################출력파일###################################
data.to_csv("C:\\future finance\\processed\\1.csv")
#################################################################################
