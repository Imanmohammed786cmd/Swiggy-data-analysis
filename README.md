**Swiggy-data-analysis**
1)**DataCleaning**
importpandasaspd

Step1:Loadthedataset
df=pd.read_csv('swiggy_data.csv')Userawstring(r)forfilepath

Step2:Dropduplicaterows
df=df.drop_duplicates()

Step3:Droprowswithmissingvalues
df=df.dropna()

Step4:Savethecleaneddata(withindex=Falsetoavoidwritingtheindexcolumn)
df.to_csv('Cleaned_data.csv',index=False)Ensurefileextension.csvandindex=False
print("Data_Cleaned")

2)**Dataconvertintopickelfileandencoderfile**
importpandasaspd
fromsklearn.preprocessingimportOneHotEncoder
importpickle

Loadthecleaneddata(cleaned_data.csvfile)
df=pd.read_csv('cleaned_data.csv')

Definethecategoricalcolumnstobeencoded
categorical_columns=['name','city','cuisine']

InitializetheOneHotEncoderwithsparse_output=True
encoder=OneHotEncoder(sparse_output=True)Thisreturnsasparsematrix

ApplyOne-HotEncodingtothecategoricalcolumns
encoded_data=encoder.fit_transform(df[categorical_columns])

ConvertthesparsematrixtoaDataFrame(ifneeded)
encoded_df=pd.DataFrame.sparse.from_spmatrix(encoded_data,columns=encoder.get_feature_names_out(categorical_columns))

Droptheoriginalcategoricalcolumnsandconcatenatetheencodedcolumns
df_encoded=pd.concat([df.drop(columns=categorical_columns),encoded_df],axis=1)

2.1.SavetheencoderasaPicklefile
withopen('encoder.pkl','wb')asf:
pickle.dump(encoder,f)

2.2.SavethepreprocesseddatasettoanewCSVfile
df_encoded.to_csv('encoded_data.csv',index=False)

2.3.Ensuretheindicesofcleaned_data.csvandencoded_data.csvmatch
assertdf.index.equals(df_encoded.index),"Indicesdonotmatch!Ensurematchingindicesbetweencleaned_dataandencoded_data."

3)**ClusteringorSimilarityMeasures:**
importpandasaspd
importnumpyasnp

Loadencodeddatasafely
encoded_df=pd.read_csv('encoded_data.csv',low_memory=False)

Replaceinvalidentrieslike'--','N/A',etc.,withNaN
encoded_df.replace(['--','N/A','NaN',''],np.nan,inplace=True)

Droporfillmissingvalues(fillingwith0issafeforencoding)
encoded_df.fillna(0,inplace=True)

Ensureallcolumnsarenumeric
encoded_df=encoded_df.apply(pd.to_numeric,errors='coerce').fillna(0)

Checkforanyremainingnon-numericcolumns
non_numeric_cols=encoded_df.select_dtypes(exclude=[np.number]).columns
print("Non-numericcolumns(shouldbeempty):",non_numeric_cols)

fromsklearn.metrics.pairwiseimportcosine_similarity

cleaned_df=pd.read_csv('cleaned_data.csv')

defrecommend_restaurants(user_index,top_n=5):
Computecosinesimilarity
similarity_scores=cosine_similarity([encoded_df.iloc[user_index]],encoded_df)[0]

Gettopsimilarindices(excludingitself)
similar_indices=similarity_scores.argsort()[::-1][1:top_n+1]

returncleaned_df.iloc[similar_indices]

Exampleusage
recommendations=recommend_restaurants(user_index=10,top_n=5)
print(recommendations)

4)**Final Step Streamlit**
importstreamlitasst
importpandasaspd
fromsklearn.metrics.pairwiseimportcosine_similarity

---Loadandpreprocessdata---
@st.cache_data
defload_data():
cleaned_df=pd.read_csv('cleaned_data.csv')
encoded_df=pd.read_csv('encoded_data.csv',low_memory=False)

Normalizecolumnnames
cleaned_df.columns=cleaned_df.columns.str.strip().str.lower()
encoded_df.columns=encoded_df.columns.str.strip().str.lower()

Ensureallvaluesinencoded_dfarenumeric
encoded_df=encoded_df.apply(pd.to_numeric,errors='coerce')
encoded_df=encoded_df.fillna(0)

returncleaned_df,encoded_df

cleaned_df,encoded_df=load_data()

---RecommendationEngine---
defrecommend_restaurants(user_index,top_n=5):
try:
similarity_scores=cosine_similarity(
[encoded_df.iloc[user_index]],
encoded_df
)[0]
similar_indices=similarity_scores.argsort()[::-1][1:top_n+1]
returncleaned_df.iloc[similar_indices]
exceptExceptionase:
st.error(f"Recommendationerror:{e}")
returnpd.DataFrame()

---StreamlitUI---
st.title("🍴RestaurantRecommender")
st.markdown("Thisapprecommendssimilarrestaurantsbasedonyourselection.")

Pickanyrestaurantasinput
restaurant_names=cleaned_df['name'].dropna().unique()if'name'incleaned_df.columnselsecleaned_df.index.astype(str)
selected_restaurant=st.selectbox("Selectarestaurantyoulike:",restaurant_names)

Findtheindexoftheselectedrestaurant
try:
if'name'incleaned_df.columns:
user_index=cleaned_df[cleaned_df['name']==selected_restaurant].index[0]
else:
user_index=int(selected_restaurant)
exceptExceptionase:
st.error(f"Failedtofindrestaurant:{e}")
st.stop()

Generateanddisplayrecommendations
recommendations=recommend_restaurants(user_index=user_index,top_n=5)

ifnotrecommendations.empty:
st.subheader("🍽️TopRecommendedRestaurants:")
st.dataframe(recommendations.reset_index(drop=True))
else:
st.warning("Norecommendationsfound.")
