import pandas as pd

def pipeline1(debug=False):
    
    # Read in relevant data
    df = pd.read_csv('data/train.csv')

    # Drop the useless features
    df = drop_useless_features(df)

    # Fix the features that contain non-independent values, such as categories with overlapping category values
    df = reclassify_non_independent_features(df)
    
    
    return

def drop_useless_features(df):
    df = df.drop(['Neighbourhood'],axis=1)
    return df

def reclassify_non_independent_features(df):
    # Reclassify the 'MSSubClass' feature

    # Replace 'MSSubClass' codes with their descriptions (from data_description.txt)
    MSSubClass_dict = {20:"1-STORY 1946 & NEWER ALL STYLES",30:"1-STORY 1945 & OLDER",40:"1-STORY W/FINISHED ATTIC ALL AGES",45:"1-1/2 STORY - UNFINISHED ALL AGES",50:"1-1/2 STORY FINISHED ALL AGES",60:"2-STORY 1946 & NEWER",70:"2-STORY 1945 & OLDER",75:"2-1/2 STORY ALL AGES",80:"SPLIT OR MULTI-LEVEL",85:"SPLIT FOYER",90:"DUPLEX - ALL STYLES AND AGES",120:"1-STORY PUD (Planned Unit Development) - 1946 & NEWER",150:"1-1/2 STORY PUD - ALL AGES",160:"2-STORY PUD - 1946 & NEWER",180:"PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",190:"2 FAMILY CONVERSION - ALL STYLES AND AGES"}
    df['MSSubClass'] = df['MSSubClass'].map(MSSubClass_dict)

    # Apply reclassifications
    df['MSSubClass_Age_Newer'] = df['MSSubClass'].apply(lambda x: 1 if '1946' in x else 0)
    df['MSSubClass_Age_Older'] = df['MSSubClass'].apply(lambda x: 1 if '1945' in x else 0)
    df['MSSubClass_Age_AllAges'] = df['MSSubClass'].apply(lambda x: 1 if '1945' not in x and '1946' not in x else 0)

    df['MSSubClass_NumStories_1'] = df['MSSubClass'].apply(lambda x: 1 if '1-STORY' in x else 0)
    df['MSSubClass_NumStories_1.5'] = df['MSSubClass'].apply(lambda x: 1 if '1-1/2 STORY' in x else 0)
    df['MSSubClass_NumStories_2'] = df['MSSubClass'].apply(lambda x: 1 if '2-STORY' in x else 0)
    df['MSSubClass_NumStories_2.5'] = df['MSSubClass'].apply(lambda x: 1 if '2-1/2 STORY' in x else 0)

    df['MSSubClass_FinishedAttic'] = df['MSSubClass'].apply(lambda x: 1 if 'FINISHED ATTIC' in x else 0)

    df['MSSubClass_DwellingType_SingleFamily'] = df['MSSubClass'].apply(lambda x: 1 if '1-STORY' in x or '2-STORY' in x else 0)
    df['MSSubClass_DwellingType_Duplex'] = df['MSSubClass'].apply(lambda x: 1 if 'DUPLEX' in x else 0)
    df['MSSubClass_DwellingType_PUD'] = df['MSSubClass'].apply(lambda x: 1 if 'PUD' in x else 0)
    df['MSSubClass_DwellingType_2FamilyConversion'] = df['MSSubClass'].apply(lambda x: 1 if '2 FAMILY CONVERSION' in x else 0)
    df['MSSubClass_DwellingType_Other'] = df['MSSubClass'].apply(lambda x: 1 if '1-STORY' not in x and '2-STORY' not in x and 'DUPLEX' not in x and 'PUD' not in x and '2 FAMILY CONVERSION' not in x else 0)

    df['MSSubClass_SplitLevel'] = df['MSSubClass'].apply(lambda x: 1 if 'SPLIT OR MULTI-LEVEL' in x or 'MULTILEVEL' in x else 0)
    df['MSSubClass_SplitFoyer'] = df['MSSubClass'].apply(lambda x: 1 if 'SPLIT FOYER' in x else 0)

    df['MSSubClass_Finished'] = df['MSSubClass'].apply(lambda x: 1 if 'FINISHED' in x else 0)
    df['MSSubClass_Unfinished'] = df['MSSubClass'].apply(lambda x: 1 if 'UNFINISHED' in x else 0)
    
    # Reclassify the 'MSZoning' feature
    df['MSZoning_Agriculture'] = df['MSZoning'].apply(lambda x: 1 if x=='A' else 0)
    df['MSZoning_Commercial'] = df['MSZoning'].apply(lambda x: 1 if x=='C' else 0)
    df['MSZoning_FloatingVillage'] = df['MSZoning'].apply(lambda x: 1 if x=='FV' else 0)
    df['MSZoning_Industrial'] = df['MSZoning'].apply(lambda x: 1 if x=='I' else 0)
    df['MSZoning_Residential'] = df['MSZoning'].apply(lambda x: 1 if x.isin(['FV','RH','RL','RP','RM']) else 0)
    df['MSZoning_DensityHigh'] = df['MSZoning'].apply(lambda x: 1 if x=='RH' else 0)
    df['MSZoning_DensityLow'] = df['MSZoning'].apply(lambda x: 1 if x.isin(['RL','RP']) else 0)
    df['MSZoning_DensityMedium'] = df['MSZoning'].apply(lambda x: 1 if x=='RM' else 0)
    df['MSZoning_Park'] = df['MSZoning'].apply(lambda x: 1 if x=='RP' else 0)
    
    # Reclassify the 'LotShape' feature
    df['LotShape_Regularity'] = df['LotShape'].apply(lambda x: 0 if x=="IR3" else 0.33 if x=="IR2" else 0.66 if x=="IR1" else 1)
    
    # Reclassify the 'LandContour' feature
    df['LandContour_Flatness'] = df['LandContour'].apply(lambda x: 0 if x=="Low" else 0.33 if x=="HLS" else 0.66 if x=="Bnk" else 1)

    # Reclassify the 'Utilities' feature
    df['Utilities_Electricity'] = df['Utilities'].apply(lambda x: 1 if x.isin(['AllPub','NoSewr','NoSeWa', 'ELO']) else 0)
    df['Utilities_Gas'] = df['Utilities'].apply(lambda x: 1 if x.isin(['AllPub','NoSewr','NoSeWa']) else 0)
    df['Utilities_Water'] = df['Utilities'].apply(lambda x: 1 if x.isin(['AllPub','NoSewr']) else 0)
    df['Utilities_Sewer'] = df['Utilities'].apply(lambda x: 1 if x.isin(['AllPub']) else 0)

    # Reclassify the 'LandSlope' feature
    df['LandSlope_Slope'] = df['LandSlope'].apply(lambda x: 0 if x=="Gtl" else 0.5 if x=="Mod" else 1)

    # Reclassify the 'Condition1' feature
    df['Condition1_Artery'] = df['Condition1'].apply(lambda x: 1 if x=='Artery' else 0)
    df['Condition1_Feedr'] = df['Condition1'].apply(lambda x: 1 if x=='Feedr' else 0)
    df['Condition1_Norm'] = df['Condition1'].apply(lambda x: 1 if x=='Norm' else 0)
    df['Condition1_NearNSR'] = df['Condition1'].apply(lambda x: 1 if x=='RRNn' else 0)
    df['Condition1_AdjacentNSR'] = df['Condition1'].apply(lambda x: 1 if x=='RRAn' else 0)
    df['Condition1_NearEWR'] = df['Condition1'].apply(lambda x: 1 if x=='RRNe' else 0)
    df['Condition1_AdjacentEWR'] = df['Condition1'].apply(lambda x: 1 if x=='RRAe' else 0)
    df['Condition1_NearPos'] = df['Condition1'].apply(lambda x: 1 if x=='PosN' else 0)
    df['Condition1_AdjacentPos'] = df['Condition1'].apply(lambda x: 1 if x=='PosA' else 0)

    # Reclassify the 'Condition2' feature, by simply ORing it with the 'Condition1' features
    df['Condition1_Artery'] = df['Condition2'].apply(lambda x: 1 if x=='Artery' else 0) | df['Condition1_Artery']
    df['Condition1_Feedr'] = df['Condition2'].apply(lambda x: 1 if x=='Feedr' else 0) | df['Condition1_Feedr']
    df['Condition1_Norm'] = df['Condition2'].apply(lambda x: 1 if x=='Norm' else 0) | df['Condition1_Norm']
    df['Condition1_NearNSR'] = df['Condition2'].apply(lambda x: 1 if x=='RRNn' else 0) | df['Condition1_NearNSR']
    df['Condition1_AdjacentNSR'] = df['Condition2'].apply(lambda x: 1 if x=='RRAn' else 0) | df['Condition1_AdjacentNSR']
    df['Condition1_NearEWR'] = df['Condition2'].apply(lambda x: 1 if x=='RRNe' else 0) | df['Condition1_NearEWR']
    df['Condition1_AdjacentEWR'] = df['Condition2'].apply(lambda x: 1 if x=='RRAe' else 0) | df['Condition1_AdjacentEWR']
    df['Condition1_NearPos'] = df['Condition2'].apply(lambda x: 1 if x=='PosN' else 0) | df['Condition1_NearPos']
    df['Condition1_AdjacentPos'] = df['Condition2'].apply(lambda x: 1 if x=='PosA' else 0) | df['Condition1_AdjacentPos']

    # Reclassify the 'BldgType' feature, by simply ORing it with the 'MSSubClass_DwellingType' features
    df['MSSubClass_DwellingType_SingleFamily'] = df['BldgType'].apply(lambda x: 1 if x=='1Fam' else 0) | df['MSSubClass_DwellingType_SingleFamily']
    df['MSSubClass_DwellingType_TownhouseEnd'] = df['BldgType'].apply(lambda x: 1 if x=='TwnhsE' else 0)
    df['MSSubClass_DwellingType_TownhouseInside'] = df['BldgType'].apply(lambda x: 1 if x=='TwnhsI' else 0)
    df['MSSubClass_DwellingType_Duplex'] = df['BldgType'].apply(lambda x: 1 if x=='Duplex' else 0) | df['MSSubClass_DwellingType_Duplex']
    df['MSSubClass_DwellingType_2FamilyConversion'] = df['BldgType'].apply(lambda x: 1 if x=='2FmCon' else 0) | df['MSSubClass_DwellingType_2FamilyConversion']

    # Reclassify the 'HouseStyle' feature, by simply ORing it with the 'MSSubClass_NumStories', 'MSSubClass_Finished', 'MSSubClass_Unfinished', 'MSSubClass_SplitFoyer' and 'MSSubClass_SplitLevel' features
    df['MSSubClass_NumStories_1'] = df['HouseStyle'].apply(lambda x: 1 if x=='1Story' else  0) | df['MSSubClass_NumStories_1']
    df['MSSubClass_NumStories_1.5'] = df['HouseStyle'].apply(lambda x: 1 if x=='1.5Fin' or x=='1.5Unf' else 0) | df['MSSubClass_NumStories_1.5']
    df['MSSubClass_NumStories_2'] = df['HouseStyle'].apply(lambda x: 1 if x=='2Story' else 0) | df['MSSubClass_NumStories_2']
    df['MSSubClass_NumStories_2.5'] = df['HouseStyle'].apply(lambda x: 1 if x=='2.5Fin' or x=='2.5Unf' else 0) | df['MSSubClass_NumStories_2.5']
    df['MSSubClass_Finished'] = df['HouseStyle'].apply(lambda x: 1 if x=='1.5Fin' or x=='2.5Fin' or x=='SFoyer' or x=='SLvl' else 0) | df['MSSubClass_Finished']
    df['MSSubClass_Unfinished'] = df['HouseStyle'].apply(lambda x: 1 if x=='1.5Unf' or x=='2.5Unf' else 0) | df['MSSubClass_Unfinished']
    df['MSSubClass_SplitFoyer'] = df['HouseStyle'].apply(lambda x: 1 if x=='SFoyer' else 0) | df['MSSubClass_SplitFoyer']
    df['MSSubClass_SplitLevel'] = df['HouseStyle'].apply(lambda x: 1 if x=='SLvl' else 0) | df['MSSubClass_SplitLevel']

    # Reclassify 'Exterior1st' and 'Exterior2nd' features by OR'ing them
    df['Exterior_AsbShng'] = df['Exterior1st'].apply(lambda x: 1 if x=='AsbShng' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='AsbShng' else 0)
    df['Exterior_AsphShn'] = df['Exterior1st'].apply(lambda x: 1 if x=='AsphShn' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='AsphShn' else 0)
    df['Exterior_BrkComm'] = df['Exterior1st'].apply(lambda x: 1 if x=='BrkComm' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='BrkComm' else 0)
    df['Exterior_BrkFace'] = df['Exterior1st'].apply(lambda x: 1 if x=='BrkFace' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='BrkFace' else 0)
    df['Exterior_CBlock'] = df['Exterior1st'].apply(lambda x: 1 if x=='CBlock' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='CBlock' else 0)
    df['Exterior_CemntBd'] = df['Exterior1st'].apply(lambda x: 1 if x=='CemntBd' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='CemntBd' else 0)
    df['Exterior_HdBoard'] = df['Exterior1st'].apply(lambda x: 1 if x=='HdBoard' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='HdBoard' else 0)
    df['Exterior_ImStucc'] = df['Exterior1st'].apply(lambda x: 1 if x=='ImStucc' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='ImStucc' else 0)
    df['Exterior_MetalSd'] = df['Exterior1st'].apply(lambda x: 1 if x=='MetalSd' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='MetalSd' else 0)
    df['Exterior_Plywood'] = df['Exterior1st'].apply(lambda x: 1 if x=='Plywood' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='Plywood' else 0)
    df['Exterior_Stone'] = df['Exterior1st'].apply(lambda x: 1 if x=='Stone' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='Stone' else 0)
    df['Exterior_Stucco'] = df['Exterior1st'].apply(lambda x: 1 if x=='Stucco' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='Stucco' else 0)
    df['Exterior_VinylSd'] = df['Exterior1st'].apply(lambda x: 1 if x=='VinylSd' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='VinylSd' else 0)
    df['Exterior_WdSdng'] = df['Exterior1st'].apply(lambda x: 1 if x=='Wd Sdng' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='Wd Sdng' else 0)
    df['Exterior_WdShing'] = df['Exterior1st'].apply(lambda x: 1 if x=='WdShing' else 0) | df['Exterior2nd'].apply(lambda x: 1 if x=='WdShing' else 0)

    

    # Drop all reclassified features
    
