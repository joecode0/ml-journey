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

    # Reclassify 'OverallQual' feature by converting numbers to range [0,1]
    df['OverallQual_Val'] = df['OverallQual'].apply(lambda x: (x-1)/9)

    # Reclassify 'OverallCond' feature by converting numbers to range [0,1]
    df['OverallCond_Val'] = df['OverallCond'].apply(lambda x: (x-1)/9)

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

    # Reclassify 'ExterQual' feature by converting values to numbers in the range [0,1]
    df['ExterQual_Val'] = df['ExterQual'].apply(lambda x: 0 if x=='Po' else 0.25 if x=='Fa' else 0.5 if x=='TA' else 0.75 if x=='Gd' else 1 if x=='Ex' else 0.5)

    # Reclassify 'ExterCond' feature by converting values to numbers in the range [0,1]
    df['ExterCond_Val'] = df['ExterCond'].apply(lambda x: 0 if x=='Po' else 0.25 if x=='Fa' else 0.5 if x=='TA' else 0.75 if x=='Gd' else 1 if x=='Ex' else 0.5)

    # Reclassify 'BsmtQual' feature by converting values to numbers in the range [0,1]
    df['BsmtQual_Val'] = df['BsmtQual'].apply(lambda x: 0 if x=='NA' else 0.6 if x=='Po' else 0.7 if x=='Fa' else 0.8 if x=='TA' else 0.9 if x=='Gd' else 1 if x=='Ex' else 0)

    # Reclassify 'BsmtFinType1' and 'BsmtFinType2' features by OR'ing them
    df['BsmtFinType_Unfinished'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='Unf' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='Unf' else 0)
    df['BsmtFinType_LwQ'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='LwQ' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='LwQ' else 0)
    df['BsmtFinType_Rec'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='Rec' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='Rec' else 0)
    df['BsmtFinType_BLQ'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='BLQ' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='BLQ' else 0)
    df['BsmtFinType_ALQ'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='ALQ' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='ALQ' else 0)
    df['BsmtFinType_GLQ'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='GLQ' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='GLQ' else 0)

    # Reclassify 'BsmtCond', 'BsmtExposure', 'BsmtFinType1' and 'BsmtFinType2' features by one-hot-encoding them and combining all their 'no basement' options
    # also OR 'BsmtFinType1' and 'BsmtFinType2' features
    df['BsmtCond_Ex'] = df['BsmtCond'].apply(lambda x: 1 if x=='Ex' else 0)
    df['BsmtCond_Gd'] = df['BsmtCond'].apply(lambda x: 1 if x=='Gd' else 0)
    df['BsmtCond_TA'] = df['BsmtCond'].apply(lambda x: 1 if x=='TA' else 0)
    df['BsmtCond_Fa'] = df['BsmtCond'].apply(lambda x: 1 if x=='Fa' else 0)
    df['BsmtCond_Po'] = df['BsmtCond'].apply(lambda x: 1 if x=='Po' else 0)

    df['BsmtExposure_Gd'] = df['BsmtExposure'].apply(lambda x: 1 if x=='Gd' else 0)
    df['BsmtExposure_Av'] = df['BsmtExposure'].apply(lambda x: 1 if x=='Av' else 0)
    df['BsmtExposure_Mn'] = df['BsmtExposure'].apply(lambda x: 1 if x=='Mn' else 0)
    df['BsmtExposure_No'] = df['BsmtExposure'].apply(lambda x: 1 if x=='No' else 0)

    df['BsmtFinType1_Unf'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='Unf' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='Unf' else 0)
    df['BsmtFinType1_LwQ'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='LwQ' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='LwQ' else 0)
    df['BsmtFinType1_Rec'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='Rec' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='Rec' else 0)
    df['BsmtFinType1_BLQ'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='BLQ' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='BLQ' else 0)
    df['BsmtFinType1_ALQ'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='ALQ' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='ALQ' else 0)
    df['BsmtFinType1_GLQ'] = df['BsmtFinType1'].apply(lambda x: 1 if x=='GLQ' else 0) | df['BsmtFinType2'].apply(lambda x: 1 if x=='GLQ' else 0)

    df['NoBasement'] = df['BsmtCond'].apply(lambda x: 1 if x=='NA' else 0) | df['BsmtExposure'].apply(lambda x: 1 if x=='NA' else 0) | df['BsmtFinType1'].apply(lambda x: 1 if x=='NA' else 0)

    # Reclassify 'HeatingQC' feature by converting values to numbers in the range [0,1]
    df['HeatingQC_Val'] = df['HeatingQC'].apply(lambda x: 0 if x=='Po' else 0.25 if x=='Fa' else 0.5 if x=='TA' else 0.75 if x=='Gd' else 1 if x=='Ex' else 0.5)

    # Reclassify 'CentralAir' feature by converting values to numbers in the range [0,1]
    df['CentralAir_Val'] = df['CentralAir'].apply(lambda x: 0 if x=='N' else 1 if x=='Y' else 0)

    # Reclassify 'KitchenQual' feature by converting values to numbers in the range [0,1]
    df['KitchenQual_Val'] = df['KitchenQual'].apply(lambda x: 0 if x=='Po' else 0.25 if x=='Fa' else 0.5 if x=='TA' else 0.75 if x=='Gd' else 1 if x=='Ex' else 0.5)

    # Reclassify 'Functional' feature by converting values to numbers in the range [0,1]
    df['Functional_Val'] = df['Functional'].apply(lambda x: 0 if x=='Sal' else 0.143 if x=='Sev' else 0.286 if x=='Maj2' else 0.429 if x=='Maj1' else 0.571 if x=='Mod' else 0.714 if x=='Min2' else 0.857 if x=='Min1' else 1) # 1 includes last case 'Typ'

    # Reclassify 'GarageType', 'GarageFinish', 'GarageQual' and 'GarageCond' features by one-hot-encoding them and combining all their 'no garage' options
    df['GarageType_2Types'] = df['GarageType'].apply(lambda x: 1 if x=='2Types' else 0)
    df['GarageType_Attchd'] = df['GarageType'].apply(lambda x: 1 if x=='Attchd' else 0)
    df['GarageType_Basment'] = df['GarageType'].apply(lambda x: 1 if x=='Basment' else 0)
    df['GarageType_BuiltIn'] = df['GarageType'].apply(lambda x: 1 if x=='BuiltIn' else 0)
    df['GarageType_CarPort'] = df['GarageType'].apply(lambda x: 1 if x=='CarPort' else 0)
    df['GarageType_Detchd'] = df['GarageType'].apply(lambda x: 1 if x=='Detchd' else 0)

    df['GarageFinish_Fin'] = df['GarageFinish'].apply(lambda x: 1 if x=='Fin' else 0)
    df['GarageFinish_RFn'] = df['GarageFinish'].apply(lambda x: 1 if x=='RFn' else 0)
    df['GarageFinish_Unf'] = df['GarageFinish'].apply(lambda x: 1 if x=='Unf' else 0)

    df['GarageQual_Ex'] = df['GarageQual'].apply(lambda x: 1 if x=='Ex' else 0)
    df['GarageQual_Gd'] = df['GarageQual'].apply(lambda x: 1 if x=='Gd' else 0)
    df['GarageQual_TA'] = df['GarageQual'].apply(lambda x: 1 if x=='TA' else 0)
    df['GarageQual_Fa'] = df['GarageQual'].apply(lambda x: 1 if x=='Fa' else 0)
    df['GarageQual_Po'] = df['GarageQual'].apply(lambda x: 1 if x=='Po' else 0)

    df['GarageCond_Ex'] = df['GarageCond'].apply(lambda x: 1 if x=='Ex' else 0)
    df['GarageCond_Gd'] = df['GarageCond'].apply(lambda x: 1 if x=='Gd' else 0)
    df['GarageCond_TA'] = df['GarageCond'].apply(lambda x: 1 if x=='TA' else 0)
    df['GarageCond_Fa'] = df['GarageCond'].apply(lambda x: 1 if x=='Fa' else 0)
    df['GarageCond_Po'] = df['GarageCond'].apply(lambda x: 1 if x=='Po' else 0)

    df['NoGarage'] = df['GarageType'].apply(lambda x: 1 if x=='NA' else 0) | df['GarageFinish'].apply(lambda x: 1 if x=='NA' else 0) | df['GarageQual'].apply(lambda x: 1 if x=='NA' else 0) | df['GarageCond'].apply(lambda x: 1 if x=='NA' else 0)

    # Reclassify 'PavedDrive' feature by converting values to numbers in the range [0,1]
    df['PavedDrive_Val'] = df['PavedDrive'].apply(lambda x: 0 if x=='N' else 0.5 if x=='P' else 1 if x=='Y' else 0)

    # Drop all reclassified features
    drop_cols = ['MSSubClass','MSZoning','LotShape','LandContour','Utilities','Landslope','Condition1',
                 'Condition2','BldgType','HouseStyle','OverallQual','OverallCond','Exterior1st','Exterior2nd',
                 'ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                 'HeatingQC','CentralAir','KitchenQual','Functional','GarageType','GarageFinish','GarageQual',
                 'GarageCond','PavedDrive']
    df.drop(drop_cols, axis=1, inplace=True)

    return df
    
