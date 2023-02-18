sm = MySmspec(r'model_1\PREDICT_445_11_2VAR_VELO_PICK.SMSPEC') # относительный путь к SMSPEC, файлы типа .S0001 и .UNSMRY подгружаются автоматически из той же папки
sm1 = MySmspec(r'C:\Users\I_Badryzlov\Desktop\BinReader\model_2\NETWORK_DEMO.SMSPEC') #абсолютный путь также работает
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns',10)

# get_data возвращает датафрейм со всеми данными
print(sm.get_data) 
print("="*150)
    
# get_main возвращает датафрейм пересечения списка KEYWORDS со списком WGNAMES
print(sm.get_main(['WGPR'], ['1011','1012'])) 
print("="*150)
print(sm.get_main(['GPR','GGPR','FGPR'],['FIELD','DKS1','DKS2']))

# список всех дат
dates = sm.get_all_dates 
# список всех уникальных KEYWORDS
keywords = sm.get_all_keywords
# список всех уникальных скважин
wells = sm.get_all_wells
# список всех уникальных аквиферов
aquifers = sm.get_all_aquifers 
# список всех уникальных групп
groups = sm.get_all_groups