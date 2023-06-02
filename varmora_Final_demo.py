import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from google.cloud import storage
import gcsfs
from sklearn.metrics import classification_report
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

json_path="/home/ubuntu/demo/Tiles_design/project-e3a05-88282a7bc99f.json"

fs = gcsfs.GCSFileSystem(token= json_path, project='project')
with fs.open('gs://project-e3a05.appspot.com/Design_full_demo.csv') as f:
    df = pd.read_csv(f)

df = df.drop(['NAME','URL','LENGTH', 'WIDTH', 'COLOR', 'SQ.FT.'] ,axis=1)

#firebase setup
cred = credentials.Certificate(json_path)
firebase_admin.initialize_app(cred, name = 'database')

firebase_admin.initialize_app(cred, {'databaseURL' : 'https://project-e3a05-default-rtdb.asia-southeast1.firebasedatabase.app/'})
ref = db.reference('recommendation/')
History = ref.child('History')
button_ref = ref.child('buttonValue')
recom_ref = ref.child('values')

len_to_code={"74":"04","27":"03","73":"02","29":"07","52":"05","63":"06","89":"01"}

code_to_name={"01": "MINIMAL", "02": "BOHEMIAN", "03": "ASIAN", "04": "ART_MODERNE","05" :"FRENCH_COUNTRY","06": "INDUSTRIAL","07": "COASTAL_TROPICAL"}


def progm():
    History.set({
        'user_data' : '0',
        'Wflag' : 'false'
        })
    button_ref.set({
        'Input' : 2,
        'flag': "false",
        'Dflag' : 'false'
        })
    inp_temp= ref.child("buttonValue").get()
    Dflag = (list(inp_temp.values())[0])
    history=[]
    full_history=[]
    temp=[]
    repeat=0
    texture_code={ "sofa":"1", "curtains":"1", "arm_chair":"1" , "tv":"5", "coffee_table":"5", "door":"5", "panel":"5", "wall_color":"4", "paintings":"2",  "rugs":"3"}
        
        
    '''
        1- sofa_texture(vineet)
        2- paintings
        3- rugs
        4- wall_paint
        5- royal_touche    
    '''
    inp_design= ref.child("Design").get()
    inp_design=list(inp_design.values())

    print(inp_design)
    
    '''
    art_modern=(df['ART_MODERNE'])
    asian=(inp_design[1])
    bohemian=str(df['BOHEMIAN'])
    costal_tropical=inp_design[3]
    french_country=inp_design[4]
    industrial=inp_design[5]
    minimal=inp_design[6]
    '''
    
    number_values=[74,27,73,29,52,63,89]
    
    len_to_name={str(number_values[0]):"ART_MODERNE",str(number_values[1]):"ASIAN",str(number_values[2]):"BOHEMIAN",str(number_values[3]):"COASTAL_TROPICAL",str(number_values[4]):"FRENCH_COUNTRY",str(number_values[5]):"INDUSTRIAL",str(number_values[6]):"MINIMAL"}
    #temp=[0, 10, 0, 1, 8, 10, 10]
    
    overall=[]
    check = 0
    
    
    for i in range(len(number_values)):
        
        t=[inp_design[i],len_to_code[str(number_values[i])]]
        overall.append(t)
    
    
    print("overall raw")
    print(overall)
    overall.sort()
    
    
    overall = overall[::-1]
    print("overall")
    print(overall)
    

    remove_ind = ["{}".format(index1) for index1,value1 in enumerate(overall) for index2,value2 in enumerate(value1) if value2==check]
    
    remove_ind = remove_ind[::-1]
    for i in remove_ind:
        overall.pop(int(i))
    
    
    print(overall)
    
    '''
    print(art_modern)
    print(asian)
    print(bohemian)
    print(costal_tropical)
    print(french_country)
    print(industrial)
    print(minimal)
    '''

        
    design_code=overall[0][1]
        
    sofa=1        
    curtains=4
    arm_chair=9

    tv=1
    coffee_table=3
    door=6
    panel=9


    wall_color=1

    paintings =1

    rugs=1
        

    
    sofa_str=str(1000+sofa)
    sofa_code=str(texture_code["sofa"])
    sofa_final_name=sofa_code+design_code+sofa_str[1:]
    recom_ref.update({'sofa': int(sofa_final_name)})

    arm_chair_str=str(1000+arm_chair)
    arm_chair_code=str(texture_code["arm_chair"])
    arm_chair_final_name=arm_chair_code+design_code+arm_chair_str[1:]
    recom_ref.update({'arm_chair': int(arm_chair_final_name)})

    curtains_str=str(1000+curtains)
    curtains_code=str(texture_code["curtains"])
    curtains_final_name=curtains_code+design_code+curtains_str[1:]
    recom_ref.update({'curtains': int(curtains_final_name)})

    tv_str=str(1000+tv)
    tv_code=str(texture_code["tv"])
    tv_final_name=tv_code+design_code+tv_str[1:]
    recom_ref.update({'tv': int(tv_final_name)})

    coffee_table_str=str(1000+coffee_table)
    coffee_table_code=str(texture_code["coffee_table"])
    coffee_table_final_name=coffee_table_code+design_code+coffee_table_str[1:]
    recom_ref.update({'coffee_table': int(coffee_table_final_name)})

    door_str=str(1000+door)
    door_code=str(texture_code["door"])
    door_final_name=door_code+design_code+door_str[1:]
    recom_ref.update({'door': int(door_final_name)})

    panel_str=str(1000+panel)
    panel_code=str(texture_code["panel"])
    panel_final_name=panel_code+design_code+panel_str[1:]
    recom_ref.update({'panel': int(panel_final_name)})

    wall_color_str=str(1000+wall_color)
    wall_color_code=str(texture_code["wall_color"])
    wall_color_final_name=wall_color_code+design_code+wall_color_str[1:]
    recom_ref.update({'wall_color': int(wall_color_final_name)})

    paintings_str=str(1000+paintings)
    paintings_code=str(texture_code["paintings"])
    paintings_final_name=paintings_code+design_code+paintings_str[1:]
    recom_ref.update({'paintings': int(paintings_final_name)})

    rugs_str=str(1000+rugs)
    rugs_code=str(texture_code["rugs"])
    rugs_final_name=rugs_code+design_code+rugs_str[1:]
    recom_ref.update({'rugs': int(rugs_final_name)})    
    
    loop = len(overall)-1
    i=0
    for i in range(len(overall)):
        
        

        if(Dflag == 'true'):
            progm()
            
        if(Dflag == 'true'):
            break
        name=code_to_name[str(overall[i][1])]
        print(name)
        
        design_code=str(overall[i][1])

        df_mini= df[df[name].astype(str).str.contains('1')]
        df_mini=df_mini[['SR_NO',name,'R','G','B','SERIES','INSPIRATION','FINISH','CATEGORIES','APPLICATION']]
        #print(df_mini)
        l = overall[i][1]
        #print(l)
        
        #start=df_mini['SR_NO'][index_value]
        #print(df_mini['SR_NO'])
        start=int(random.choice(list(df_mini['SR_NO'])))
        index_value=list(df_mini['SR_NO']).index(start)
        #print(index_value)
        print(start)
        y=df_mini.iloc[:,0]
        x=df_mini.iloc[:,2:]
        while True:
            #print(index_value)
            if( Dflag == 'true'):
                break
            inp_temp= ref.child("buttonValue").get()
            Dflag = (list(inp_temp.values())[0])
            inp = (list(inp_temp.values())[1])
            flag = (list(inp_temp.values())[2])
            
            
            if(inp!= 5 and flag == "false"):
                recom_ref.update({'initial': int(start)})
                if start not in history:
                    print("adding new element: ", )
                    history.append(start)
                    print(history)
            
            
            
            if(inp==5 and flag == "true"):
                button_ref.update({
                    'flag': "false"
                    })
                full_history=full_history+history
                hist = ', '.join(str(e) for e in full_history)
                History.set({
                    'user_data' : hist,
                    'Wflag' : "true"
                    })
                
                print(full_history)
                del full_history[:]
                del history[:]
    
            elif(inp==0 and flag == "true"): #inp == 0 and flag == True
                
                print('\n \n start value: ', start)

                history.pop()
                button_ref.update({'flag': "false"})

                sofa=1        
                curtains=4
                arm_chair=9

                tv=1
                coffee_table=3
                door=6
                panel=9


                wall_color=1

                paintings =1

                rugs=1
                

                if(i+1>len(overall)-1):
                    i=-1
                design_code=str(overall[i+1][1])
                sofa_str=str(1000+sofa)
                sofa_code=str(texture_code["sofa"])
                sofa_final_name=sofa_code+design_code+sofa_str[1:]
                recom_ref.update({'sofa': int(sofa_final_name)})

                arm_chair_str=str(1000+arm_chair)
                arm_chair_code=str(texture_code["arm_chair"])
                arm_chair_final_name=arm_chair_code+design_code+arm_chair_str[1:]
                recom_ref.update({'arm_chair': int(arm_chair_final_name)})

                curtains_str=str(1000+curtains)
                curtains_code=str(texture_code["curtains"])
                curtains_final_name=curtains_code+design_code+curtains_str[1:]
                recom_ref.update({'curtains': int(curtains_final_name)})

                tv_str=str(1000+tv)
                tv_code=str(texture_code["tv"])
                tv_final_name=tv_code+design_code+tv_str[1:]
                recom_ref.update({'tv': int(tv_final_name)})

                coffee_table_str=str(1000+coffee_table)
                coffee_table_code=str(texture_code["coffee_table"])
                coffee_table_final_name=coffee_table_code+design_code+coffee_table_str[1:]
                recom_ref.update({'coffee_table': int(coffee_table_final_name)})

                door_str=str(1000+door)
                door_code=str(texture_code["door"])
                door_final_name=door_code+design_code+door_str[1:]
                recom_ref.update({'door': int(door_final_name)})

                panel_str=str(1000+panel)
                panel_code=str(texture_code["panel"])
                panel_final_name=panel_code+design_code+panel_str[1:]
                recom_ref.update({'panel': int(panel_final_name)})

                wall_color_str=str(1000+wall_color)
                wall_color_code=str(texture_code["wall_color"])
                wall_color_final_name=wall_color_code+design_code+wall_color_str[1:]
                recom_ref.update({'wall_color': int(wall_color_final_name)})

                paintings_str=str(1000+paintings)
                paintings_code=str(texture_code["paintings"])
                paintings_final_name=paintings_code+design_code+paintings_str[1:]
                recom_ref.update({'paintings': int(paintings_final_name)})

                rugs_str=str(1000+rugs)
                rugs_code=str(texture_code["rugs"])
                rugs_final_name=rugs_code+design_code+rugs_str[1:]
                recom_ref.update({'rugs': int(rugs_final_name)})
                

                break
                
                
            elif(inp==1 and flag == "true"): #inp == 1 and flag == True
                
                sofa+=1
                if(sofa>10):
                    sofa-=10
                sofa_str=str(1000+sofa)
                sofa_code=str(texture_code["sofa"])
                sofa_final_name=sofa_code+design_code+sofa_str[1:]
                recom_ref.update({'sofa': int(sofa_final_name)})

                arm_chair+=1
                if(arm_chair>10):
                    arm_chair-=10
                arm_chair_str=str(1000+arm_chair)
                arm_chair_code=str(texture_code["arm_chair"])
                arm_chair_final_name=arm_chair_code+design_code+arm_chair_str[1:]
                recom_ref.update({'arm_chair': int(arm_chair_final_name)})

                curtains+=1
                if(curtains>10):
                    curtains-=10
                curtains_str=str(1000+curtains)
                curtains_code=str(texture_code["curtains"])
                curtains_final_name=curtains_code+design_code+curtains_str[1:]
                recom_ref.update({'curtains': int(curtains_final_name)})

                tv+=1
                if(tv>10):
                    tv-=10
                tv_str=str(1000+tv)
                tv_code=str(texture_code["tv"])
                tv_final_name=tv_code+design_code+tv_str[1:]
                recom_ref.update({'tv': int(tv_final_name)})

                coffee_table+=1
                if(coffee_table>10):
                    coffee_table-=10
                coffee_table_str=str(1000+coffee_table)
                coffee_table_code=str(texture_code["coffee_table"])
                coffee_table_final_name=coffee_table_code+design_code+coffee_table_str[1:]
                recom_ref.update({'coffee_table': int(coffee_table_final_name)})

                door+=1
                if(door>10):
                    door-=10
                door_str=str(1000+door)
                door_code=str(texture_code["door"])
                door_final_name=door_code+design_code+door_str[1:]
                recom_ref.update({'door': int(door_final_name)})

                panel+=1
                if(panel>10):
                    panel-=10  
                panel_str=str(1000+panel)
                panel_code=str(texture_code["panel"])
                panel_final_name=panel_code+design_code+panel_str[1:]
                recom_ref.update({'panel': int(panel_final_name)})

                wall_color+=1
                if(wall_color>10):
                    wall_color-=10
                wall_color_str=str(1000+wall_color)
                wall_color_code=str(texture_code["wall_color"])
                wall_color_final_name=wall_color_code+design_code+wall_color_str[1:]
                recom_ref.update({'wall_color': int(wall_color_final_name)})

                paintings+=1
                if(paintings>10):
                    paintings-=10

                paintings_str=str(1000+paintings)
                paintings_code=str(texture_code["paintings"])
                paintings_final_name=paintings_code+design_code+paintings_str[1:]
                recom_ref.update({'paintings': int(paintings_final_name)})

                rugs+=1
                if(rugs>10):
                    rugs-=10
                rugs_str=str(1000+rugs)
                rugs_code=str(texture_code["rugs"])
                rugs_final_name=rugs_code+design_code+rugs_str[1:]
                recom_ref.update({'rugs': int(rugs_final_name)})

                
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=int(random.random()*100))
                
                try:
                    if ( x_train.loc[start].any()):
                        x_train=x_train.drop(index_value,axis=0)
                        y_train=y_train.drop(index_value,axis=0)
                except:
                    present=0
    
                y_train=y_train.values
                
                
                input_pred=df_mini.iloc[index_value][2:]
                input_pred=np.array(input_pred)
                input_pred=input_pred.reshape(1,-1)
    
                
                knn_model = KNeighborsClassifier(n_neighbors =3)
                knn_model.fit(x_train,y_train)
                
                
                output=knn_model.predict(input_pred)
                #print(classification_report(y_test, output))
                output=output[0]
                #output=df['SR NO'][output]
                output2=0
                if(output not in history and start != output):
                    start=output   
                else: 
                    repeat+=1
                    input_pred=df_mini.iloc[index_value][2:]
                    input_pred=np.array(input_pred)
                    input_pred=input_pred.reshape(1,-1)
                    output2=knn_model.predict(input_pred)
                    output2=output2[0]
                    #output2=df_mini['SR_NO'][output2]
                    start=output2
                    print('This is value  of output2', start)
                    
                if(repeat>=1 and output2==output):
                    
                    full_history=full_history+history
                    history=[]
                    repeat=0
                    start=int(random.choice(list(df_mini['SR_NO'])))
                    index_value=list(df_mini['SR_NO']).index(start)
                    print('\nThis is value  of random start value: ', start)
    
                #recom_ref.update({'initial': int(prev)})
                #recom_ref.update({'recommendation': int(start)})
                button_ref.update({
                    'flag': "false"
                    })
                print("\n\n Start value: ", start)
                print(history)
    
                #index_value=list(df2['SR NO']).index(start)
    
                index_value=list(df_mini['SR_NO']).index(start)
                
            
        loop -= 1

        
        if(Dflag == 'true' or loop <= -1):
            progm()
    
progm()
