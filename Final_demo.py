import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from google.cloud import storage
import gcsfs
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db



json_path="/home/ubuntu/demo/Tiles_design/project-e3a05-88282a7bc99f.json"

fs = gcsfs.GCSFileSystem(token= json_path, project='project')
with fs.open('gs://project-e3a05.appspot.com/tile_designdata.csv') as f:
    df = pd.read_csv(f)

df = df.drop(['NAME','URL','LENGTH', 'WIDTH', 'COLOR', 'SQ.FT.'] ,axis=1)

#firebase setup
cred = credentials.Certificate(json_path)
firebase_admin.initialize_app(cred, name = 'database')

#firebase_admin.initialize_app(cred, {'databaseURL' : 'https://project-e3a05-default-rtdb.asia-southeast1.firebasedatabase.app/'})

firebase_admin.initialize_app(cred, {'databaseURL' : 'https://project-e3a05-default-rtdb.asia-southeast1.firebasedatabase.app/'})
ref = db.reference('recommendation/')
button_ref = ref.child('buttonValue')
recom_ref = ref.child('values')


def progm():
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
    
    
    inp_design= ref.child("Design").get()
    inp_design=list(inp_design.values())
    sorted_design_arr=inp_design
    
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
    
    number_values=[40,9,32,19,13,24,39]
    
    len_to_name={"40":"ART_MODERNE","9":"ASIAN","32":"BOHEMIAN","19":"COASTAL_TROPICAL","13":"FRENCH_COUNTRY","24":"INDUSTRIAL","39":"MINIMAL"}
    #temp=[0, 10, 0, 1, 8, 10, 10]
    overall=[]
    
    
    for i in range(len(number_values)):
        
        t=[inp_design[i],number_values[i]]
        overall.append(t)
    
    print(overall)
    overall.sort()
    
    
    overall = overall[::-1]
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
    
    
    for i in overall:
        
        if(Dflag == 'true'):
            progm()
            
        if(Dflag == 'true'):
            break
        name=len_to_name[str(i[1])]
        print(name)
    
        df_mini= df[df[name].astype(str).str.contains('1')]
        df_mini=df_mini[['SR_NO',name,'R','G','B','SERIES','INSPIRATION','FINISH','CATEGORIES','APPLICATION']]
        print(df_mini)
        l = i[1]
        print(l)
        
        #start=df_mini['SR_NO'][index_value]
        print(df_mini['SR_NO'])
        start=int(random.choice(list(df_mini['SR_NO'])))
        index_value=list(df_mini['SR_NO']).index(start)
        print(index_value)
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
            present=0
            prev=index_value
            
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
                print(full_history)
                del history[:]
    
            elif(inp==0 and flag == "true"): #inp == 0 and flag == True
                print('\n \n start value: ', start)
                #history.pop()
                button_ref.update({'flag': "false"})
                break
                
                
            elif(inp==1 and flag == "true"): #inp == 1 and flag == True
                
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
                recom_ref.update({'recommendation': int(start)})
                button_ref.update({
                    'flag': "false"
                    })
                print("\n\n Start value: ", start)
                print(history)
    
                #index_value=list(df2['SR NO']).index(start)
    
                index_value=list(df_mini['SR_NO']).index(start)
        if(Dflag == 'true'):
            progm()
    
progm()
