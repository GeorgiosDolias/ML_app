import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')

#---------------------------------#
# Model building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
    
    scaler = StandardScaler()

    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)


    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    svr = SVR(kernel = parameter_kernel,
            degree = parameter_degree,
            gamma = parameter_gamma,
            coef0 = parameter_coef0,
            tol = parameter_tol,
            C = parameter_C,
            epsilon = parameter_epsilon,
            shrinking = parameter_shrinking,
            cache_size = parameter_cache_size,
            verbose = parameter_verbose,
            max_iter = parameter_max_iter)



    svr.fit(scaled_X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = svr.predict(scaled_X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = svr.predict(scaled_X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(svr.get_params())

#---------------------------------#
st.write("""
# The Machine Learning App

In this app, the *SVR()* function is used in order to build a regression model using the **Support Vector Regression** algorithm.

You can cheeck more details about this function here: [Sklearn's SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

Try adjusting the hyperparameters!

""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_kernel = st.sidebar.select_slider('Type of kernel (kernel)', options=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
    parameter_C = st.sidebar.slider('Regularization parameter. Regularization\'s strength is inversely proportional to C (strictly positive) ', 1, 10, 2, 1)
    parameter_epsilon = st.sidebar.slider('Epsilon in the epsilon-SVR model.', 1, 10, 2, 1)
    parameter_degree = st.sidebar.slider('Degree of the polynomial kernel function (‘poly’)', 1,6,3,1)
    parameter_gamma = st.sidebar.slider('Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid', 1, 10, 2, 1)
    parameter_coef0 = st.sidebar.slider('Independent term in kernel function (only significant in ‘poly’ and ‘sigmoid)', 1, 10, 2, 1)
    
    

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_max_iter = st.sidebar.slider('Hard limit on iterations within solver (-1 for no limit)', 1, 15000, 40, 1)
    parameter_shrinking = st.sidebar.select_slider('Whether to use the shrinking heuristic.',options=[True, False])
    parameter_cache_size = st.sidebar.slider('Specify the size of the kernel cache', 1, 10, 2, 1)
    parameter_verbose = st.sidebar.select_slider('Enable verbose output.', options=[True, False])
    parameter_tol = st.sidebar.slider('Tolerance for stopping criterion.', 1, 10, 2, 1)

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        
        # Cement Slump dataset        
        
        CWD = os.path.abspath('.')

        folderpath_processed_data = CWD + '/cement_slump.csv'

        df = pd.read_csv(folderpath_processed_data)

        st.markdown('The Cement Slump dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)