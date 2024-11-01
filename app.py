from flask import Flask, render_template, session, url_for, redirect,request, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, SubmitField, FloatField, IntegerField
from tensorflow.keras.models import load_model
from joblib import load
from sklearn.preprocessing import OneHotEncoder
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms.validators import DataRequired
from sklearn.metrics import confusion_matrix,classification_report
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import os
import logging
import hashlib


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app=Flask(__name__)
#logging.basicConfig(filename='app.log', level=logging.DEBUG)
# On s'assure que la personne qui rempli le formulaire est la même qui consulte les résultats
app.config['SECRET_KEY']='mysecretkeyHacktheworld130@'
app.config['UPLOAD_FOLDER'] = 'upload/'
save_folder='save/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

#Initialisation de Flask-Login
login_manager=LoginManager()
login_manager.init_app(app)
# chargement des modèles
model_NoN=load_model('MNoN.h5')
model_tcp=load_model('MTcp.h5')
model_udp=load_model('MUdp.h5')
model_icmp=load_model('MIcmp.h5')
model_rose=load_model('MRose.h5')

# chargement des scalers
scaler_NoN=load('scaler_MNoN.pkl')
scaler_tcp=load('scaler_MTcp.pkl')
scaler_udp=load('scaler_MUdp.pkl')
scaler_icmp=load('scaler_MIcmp.pkl')
scaler_rose=load('scaler_MRose.pkl')
### Pour la prédiction sur un paquet
def return_prediction(model,scaler,sample_json,is_protocol):
    duration=sample_json['duration']
    protocol_type=sample_json['protocol_type']
    service=sample_json['service']
    flag=sample_json['flag']
    src_bytes=sample_json['src_bytes']
    dst_bytes=sample_json['dst_bytes']
    land=sample_json['land']
    wrong_fragment=sample_json['wrong_fragment']
    urgent=sample_json['urgent']
    hot=sample_json['hot']
    num_failed_logins=sample_json['num_failed_logins']
    logged_in=sample_json['logged_in']
    num_compromised=sample_json['num_compromised']
    root_shell=sample_json['root_shell']
    su_attempted=sample_json['su_attempted']
    num_root=sample_json['num_root']
    num_file_creations=sample_json['num_file_creations']
    num_shells=sample_json['num_shells']
    num_access_files=sample_json['num_access_files']
    is_host_login=sample_json['is_host_login']
    is_guest_login=sample_json['is_guest_login']
    count=sample_json['count']
    srv_count=sample_json['srv_count']
    serror_rate=sample_json['serror_rate']
    srv_serror_rate=sample_json['srv_serror_rate']
    rerror_rate=sample_json['rerror_rate']
    srv_rerror_rate=sample_json['srv_rerror_rate']
    same_srv_rate=sample_json['same_srv_rate']
    diff_srv_rate=sample_json['diff_srv_rate']
    srv_diff_host_rate=sample_json['srv_diff_host_rate']
    dst_host_count=sample_json['dst_host_count']
    dst_host_srv_count=sample_json['dst_host_srv_count']
    dst_host_same_srv_rate=sample_json['dst_host_same_srv_rate']
    dst_host_diff_srv_rate=sample_json['dst_host_diff_srv_rate']
    dst_host_same_src_port_rate=sample_json['dst_host_same_src_port_rate']
    dst_host_srv_diff_host_rate=sample_json['dst_host_srv_diff_host_rate']
    dst_host_serror_rate=sample_json['dst_host_serror_rate']
    dst_host_srv_serror_rate=sample_json['dst_host_srv_serror_rate']
    dst_host_rerror_rate=sample_json['dst_host_rerror_rate']
    dst_host_srv_rerror_rate=sample_json['dst_host_srv_rerror_rate']

    paquet=[[duration, protocol_type, service, flag, src_bytes,dst_bytes, land, wrong_fragment, urgent, hot,
           num_failed_logins, logged_in, num_compromised, root_shell, su_attempted,num_root, num_file_creations,num_shells,
           num_access_files,is_host_login, is_guest_login, count, srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate, same_srv_rate, 
           diff_srv_rate,srv_diff_host_rate,dst_host_count, dst_host_srv_count,dst_host_same_srv_rate, dst_host_diff_srv_rate,dst_host_same_src_port_rate,
           dst_host_srv_diff_host_rate,dst_host_serror_rate, dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate]]

    classes=['normal','anormal']
    paquet=scaler.transform(paquet)

    if(is_protocol==2):
        classes=['Unauthorized access','Denial of services','Port scanning']
        class_index=np.argmax(model.predict(paquet),axis=-1)[0]
        print("La classe pour Rose est: ",class_index)
    else:
       class_index=int((model.predict(paquet)>0.5).astype('int32'))

    return classes[class_index]

### Pour la prédiction sur un dataset



# Les fonctions de traitement des données
def normal_or_not(state):
    if(state=="normal"):
        return 0
    else:
        return 1

def Maria_dataset(df,model_NoN,model_tcp,model_udp,model_icmp,predictions_maria,predictions_protocol,predictions_NoN,maria_classification_report):
    predictions_udp=[]
    predictions_tcp=[]
    predictions_icmp=[]
    df_Maria=df.copy()
    df_Maria['attack-type']=df['attack-type'].apply(normal_or_not)
    y=df_Maria['attack-type'].values
    ## Pour NoN
    df_NoN=df_Maria.copy()
    df_NoN=df_NoN.drop('level',axis=1)
    features=df_NoN.drop('attack-type',axis=1).values
    features=scaler_NoN.transform(features) # Normalisation des données 
    predictions=(model_NoN.predict(features)>0.5).astype('int32') # Faire une prédiction avec le modèle
    for element in predictions:
        predictions_NoN.append(element[0])
    #predictions_NoN.append(element[0] for element in predictions)
    # predictions_NoN.append(make_predictions_0_1(model_NoN,scaler_NoN,X_NoN))
    # Prétraitement
    ## Pour tcp
    df_tcp=df_Maria.copy()
    df_tcp=df_tcp[df_tcp['protocol_type']==1]
    index_tcp=list(df_tcp.index)
    df_tcp=df_tcp.drop('level',axis=1)
    features=df_tcp.drop('attack-type',axis=1).values
    features=scaler_tcp.transform(features) # Normalisation des données 
    prediction_tcp=(model_tcp.predict(features)>0.5).astype('int32') # Faire une prédiction avec le modèle
    predictions_tcp=list(int(element) for element in prediction_tcp)

    # Pour udp
    df_udp=df_Maria.copy()
    df_udp=df_udp[df_udp['protocol_type']==2]
    index_udp=list(df_udp.index)
    df_udp=df_udp.drop('level',axis=1)
    features=df_udp.drop('attack-type',axis=1).values
    features=scaler_udp.transform(features) # Normalisation des données 
    prediction_udp=(model_udp.predict(features)>0.5).astype('int32') # Faire une prédiction avec le modèle
    predictions_udp=list(int(element) for element in prediction_udp)
    
    # Pour icmp
    df_icmp=df_Maria.copy()
    df_icmp=df_icmp[df_icmp['protocol_type']==3]
    index_icmp=list(df_icmp.index)
    df_icmp=df_icmp.drop('level',axis=1)
    features=df_icmp.drop('attack-type',axis=1).values
    features=scaler_icmp.transform(features) # Normalisation des données 
    prediction_icmp=(model_icmp.predict(features)>0.5).astype('int32') # Faire une prédiction avec le modèle
    predictions_icmp=list(int(element) for element in prediction_icmp)

    i=0
    j=0
    k=0
    liste=list(df_Maria.index)
    for index in liste:
        if (index in index_tcp):
            predictions_protocol.append(predictions_tcp[i])
            predictions_maria.append(A_int(predictions_NoN[index],predictions_tcp[i],1))
            i=i+1
        elif(index in index_udp):
            predictions_protocol.append(predictions_udp[j])
            predictions_maria.append(A_int(predictions_NoN[index],predictions_udp[j],2))
            j=j+1
        elif(index in index_icmp):
            predictions_protocol.append(predictions_icmp[k])
            predictions_maria.append(A_int(predictions_NoN[index],predictions_icmp[k],3))
            k=k+1
    
            
def Rose(df,liste_anormal, model_rose, scaler_rose, predictions_rose):
    # Prétraitement
    dico_classement_attaque={'neptune':2,'smurf':2,'apache2':2,'back':1,'processtable':2,'pod':2,'mailbomb':2,'land':2,'teardrop':2,
                          'udpstorm':2,'saint':3,'mscan':3,'satan':3,'nmap':3,'ipsweep':3,'portsweep':3,'guess_passwd':1,'snmpgetattack':1,
                          'snmpguess':1,'ftp_write':1,'imap':1,'phf':1,'warezclient':1,'spy':1,'buffer_overflow':1,'httptunnel':1,'ps':1,'multihop':1,
                          'named':1,'sendmail':1,'loadmodule':1,'xterm':1,'worm':1,'rootkit':1,'xlock':1,'perl':1,'xsnoop':1,'sqlattack':1,'warezmaster':3}
    df_rose=df.iloc[liste_anormal].copy()
 
    df_rose['attack-type']=df_rose['attack-type'].map(dico_classement_attaque)
    df_rose.to_csv("rose_df.csv",index=False)
    df_rose=df_rose.drop(['level'],axis=1)
    
    X_rose=df_rose.drop('attack-type',axis=1).values
    y_rose=df_rose['attack-type'].values
    X_rose=scaler_rose.transform(X_rose)
    y_rose=y_rose[:].reshape(-1,1)
    encoder= OneHotEncoder(sparse_output=False)
    y_rose_encoded=encoder.fit_transform(y_rose)
    #y_rose_encoded=np.int32(y_rose)
    # Predictions pour le modèle rose
    ##predictions=(model_rose.predict(X_rose)>0.5).astype('int32')
    predictions=model_rose.predict(X_rose)
    predictions_middle=np.argmax(predictions, axis=1)
    for i in range(0,len(predictions_middle)):
    	predictions_rose.append(predictions_middle[i]+1)

# fonction de hashage du mot de passe
def hash_password(password):
    return  hashlib.sha256(password.encode('utf-8')).hexdigest()

# fonction de vérification du mot de passe
def verify_password(password, hashed_password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()== hashed_password

# classe utilisateur
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    date_registered = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
#chargement des utilisateurs
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class PaquetForm(FlaskForm):
    duration=StringField('duration', validators=[DataRequired()])
    protocol_type=StringField('protocol_type', validators=[DataRequired()])
    service=StringField('service', validators=[DataRequired()])
    flag=StringField('flag', validators=[DataRequired()])
    src_bytes=StringField('src_bytes', validators=[DataRequired()])
    dst_bytes=StringField('dst_bytes', validators=[DataRequired()])
    land=StringField('land', validators=[DataRequired()])
    wrong_fragment=StringField('wrong_fragment', validators=[DataRequired()])
    urgent=StringField('urgent', validators=[DataRequired()])
    hot=StringField('hot', validators=[DataRequired()])
    num_failed_logins=StringField('num_failed_logins', validators=[DataRequired()])
    logged_in=StringField('logged_in', validators=[DataRequired()])
    num_compromised=StringField('num_compromised', validators=[DataRequired()])
    root_shell=StringField('root_shell', validators=[DataRequired()])
    su_attempted=StringField('su_attempted', validators=[DataRequired()])
    num_root=StringField('num_root', validators=[DataRequired()])
    num_file_creations=StringField('num_file_creations', validators=[DataRequired()])
    num_shells=StringField('num_shells', validators=[DataRequired()])
    num_access_files=StringField('num_access_files', validators=[DataRequired()])
    is_host_login=StringField('is_host_login', validators=[DataRequired()])
    is_guest_login=StringField('is_guest_login', validators=[DataRequired()])
    count=StringField('count', validators=[DataRequired()])
    srv_count=StringField('srv_count', validators=[DataRequired()])
    serror_rate=StringField('serror_rate', validators=[DataRequired()])
    srv_serror_rate=StringField('srv_serror_rate', validators=[DataRequired()])
    rerror_rate=StringField('rerror_rate', validators=[DataRequired()])
    srv_rerror_rate=StringField('srv_rerror_rate', validators=[DataRequired()])
    same_srv_rate=StringField('same_srv_rate', validators=[DataRequired()])
    diff_srv_rate=StringField('diff_srv_rate', validators=[DataRequired()])
    srv_diff_host_rate=StringField('srv_diff_host_rate', validators=[DataRequired()])
    dst_host_count=StringField('dst_host_count', validators=[DataRequired()])
    dst_host_srv_count=StringField('dst_host_srv_count', validators=[DataRequired()])
    dst_host_same_srv_rate=StringField('dst_host_same_srv_rate', validators=[DataRequired()])
    dst_host_diff_srv_rate=StringField('dst_host_diff_srv_rate', validators=[DataRequired()])
    dst_host_same_src_port_rate=StringField('dst_host_same_src_port_rate', validators=[DataRequired()])
    dst_host_srv_diff_host_rate=StringField('dst_host_srv_diff_host_rate', validators=[DataRequired()])
    dst_host_serror_rate=StringField('dst_host_serror_rate', validators=[DataRequired()])
    dst_host_srv_serror_rate=StringField('dst_host_srv_serror_rate', validators=[DataRequired()])
    dst_host_rerror_rate=StringField('dst_host_rerror_rate', validators=[DataRequired()])
    dst_host_srv_rerror_rate=StringField('dst_host_srv_rerror_rate', validators=[DataRequired()])
    
    submit=SubmitField('Analyser')

class FileForm(FlaskForm):
    file = FileField('Fichier', validators=[FileRequired(), FileAllowed(['csv','txt'], 'Seuls les fichiers CSV et TXT sont autorisés.')])
    submit = SubmitField('Analyser le Fichier')
    

@app.route("/")
def home():
    #app.logger.debug('Ceci est un message de débogage.')
    active_page='home'
    return render_template('index.html',active_page=active_page)
	
@app.route('/paquet_prediction', methods=['GET','POST'])
def paquet_prediction():
    content={}
    if request.method =='POST':
        content['duration']=int(request.form['duration'])
        content['protocol_type']=int(request.form['protocol_type'])
        content['service']=int(request.form['service'])
        content['flag']=int(request.form['flag'])
        content['src_bytes']=int(request.form['src_bytes'])
        content['dst_bytes']=int(request.form['dst_bytes'])
        content['land']=int(request.form['land'])
        content['wrong_fragment']=int(request.form['wrong_fragment'])    
        content['urgent']=int(request.form['urgent'])    
        content['hot']=int(request.form['hot'])
        content['num_failed_logins']=int(request.form['num_failed_logins'])    
        content['logged_in']=int(request.form['logged_in'])
        content['num_compromised']=int(request.form['num_compromised'])
        content['root_shell']=int(request.form['root_shell'])
        content['num_root']=int(request.form['num_root'])
        content['su_attempted']=int(request.form['su_attempted'])
        content['num_file_creations']=int(request.form['num_file_creations'])    
        content['num_shells']=int(request.form['num_shells'])
        content['num_access_files']=int(request.form['num_access_files'])    
        content['is_host_login']=int(request.form['is_host_login'])
        content['is_guest_login']=int(request.form['is_guest_login'])    
        content['count']=int(request.form['count'])
        content['srv_count']=int(request.form['srv_count'])
        content['serror_rate']=float(request.form['serror_rate'])
        content['srv_serror_rate']=float(request.form['srv_serror_rate'])
        content['rerror_rate']=float(request.form['rerror_rate'])
        content['srv_rerror_rate']=float(request.form['srv_rerror_rate'])    
        content['same_srv_rate']=float(request.form['same_srv_rate'])    
        content['diff_srv_rate']=float(request.form['diff_srv_rate'])
        content['srv_diff_host_rate']=float(request.form['srv_diff_host_rate'])    
        content['dst_host_count']=int(request.form['dst_host_count'])
        content['dst_host_srv_count']=int(request.form['dst_host_srv_count'])
        content['dst_host_same_srv_rate']=float(request.form['dst_host_same_srv_rate'])
        content['dst_host_diff_srv_rate']=float(request.form['dst_host_diff_srv_rate'])
        content['dst_host_same_src_port_rate']=float(request.form['dst_host_same_src_port_rate'])
        content['dst_host_srv_diff_host_rate']=float(request.form['dst_host_srv_diff_host_rate'])
        content['dst_host_serror_rate']=float(request.form['dst_host_serror_rate'])
        content['dst_host_srv_serror_rate']=float(request.form['dst_host_srv_serror_rate'])
        content['dst_host_rerror_rate']=float(request.form['dst_host_rerror_rate'])
        content['dst_host_srv_rerror_rate']=float(request.form['dst_host_srv_rerror_rate'])
        
        print(content)
        result_maria="normal"
        result_protocole=""
        result_rose=""
        result_NoN=""
        result_NoN=return_prediction(model_NoN,scaler_NoN,content,0)
        print(result_NoN)
        if(content['protocol_type']==1):
            result_protocole=return_prediction(model_tcp,scaler_tcp,content,1)
            type_protocole=1
        elif(content['protocol_type']==2):
            result_protocole=return_prediction(model_udp,scaler_udp,content,1)
            type_protocole=2
        elif(content['protocol_type']==3):
            result_protocole=return_prediction(model_icmp,scaler_icmp,content,1)
            type_protocole=3
            
        #if(result_NoN=="anormal" and result_protocole=="anormal"):
         #   result_maria="anormal"
        result_maria=A(result_NoN,result_protocole,type_protocole)
        
        if(result_maria=="anormal"):
            print("ici")
            result_rose=return_prediction(model_rose,scaler_rose,content,2)

        return render_template('paquet_prediction.html',result_NoN=result_NoN,result_protocole=result_protocole,result_rose=result_rose)

    return render_template('no_result.html')

def A(result_NoN,result_protocole,type_protocole):
    dico={"normal":0,"anormal":1}
    dico_r={0:"normal",1:"anormal"}
    α=1
    if(type_protocole==1):
        α=0.3
    elif(type_protocole==2):
        α=1
    else:
        α=0.6
        
    result=α*dico[result_NoN]+(1-α)*dico[result_protocole]

    if(result>=0.5):
        result=1
    else:
        result=0
        
    return dico_r[result]

def A_int(result_NoN,result_protocole,type_protocole):
    dico={"normal":0,"anormal":1}
    α=1
    if(type_protocole==1):
        α=0.3
    elif(type_protocole==2):
        α=0.4
        if(result_protocole==1):
            α=0.6
        
    else:
        α=0.6
        if(result_protocole==1):
            α=0.4
        
    result=α*result_NoN+(1-α)*result_protocole

    if(result>=0.5):
        result=1
    else:
        result=0
        
    return result	
@app.route("/paquet_analyse", methods=['GET','POST'])
def paquet_analyse():
	form=PaquetForm()	
	if form.validate_on_submit():# s'assurer que le formulaire contient des données
		session['duration']=form.duration.data
		session['protocol_type']=form.protocol_type.data
		session['service']=form.service.data
		session['flag']=form.flag.data
		session['src_bytes']=form.src_bytes.data
		session['dst_bytes']=form.dst_bytes.data
		session['land']=form.land.data
		session['wrong_fragment']=form.wrong_fragment.data
		session['urgent']=form.urgent.data
		session['hot']=form.hot.data
		session['num_failed_logins']=form.num_failed_logins.data
		session['logged_in']=form.logged_in.data
		session['num_compromised']=form.num_compromised.data
		session['root_shell']=form.root_shell.data
		session['su_attempted']=form.su_attempted.data
		session['num_root']=form.num_root.data
		session['num_file_creations']=form.num_file_creations.data
		session['num_shells']=form.num_shells.data
		session['num_access_files']=form.num_access_files.data
		session['is_host_login']=form.is_host_login.data
		session['is_guest_login']=form.is_guest_login.data
		session['count']=form.count.data
		session['srv_count']=form.srv_count.data
		session['serror_rate']=form.serror_rate.data
		session['srv_serror_rate']=form.srv_serror_rate.data
		session['rerror_rate']=form.rerror_rate.data
		session['srv_rerror_rate']=form.srv_rerror_rate.data
		session['same_srv_rate']=form.same_srv_rate.data
		session['diff_srv_rate']=form.diff_srv_rate.data
		session['srv_diff_host_rate']=form.srv_diff_host_rate.data
		session['dst_host_count']=form.dst_host_count.data
		session['dst_host_srv_count']=form.dst_host_srv_count.data
		session['dst_host_same_srv_rate']=form.dst_host_same_srv_rate.data
		session['dst_host_diff_srv_rate']=form.dst_host_diff_srv_rate.data
		session['dst_host_same_src_port_rate']=form.dst_host_same_src_port_rate.data
		session['dst_host_srv_diff_host_rate']=form.dst_host_srv_diff_host_rate.data
		session['dst_host_serror_rate']=form.dst_host_serror_rate.data
		session['dst_host_srv_serror_rate']=form.dst_host_srv_serror_rate.data
		session['dst_host_rerror_rate']=form.dst_host_rerror_rate.data
		session['dst_host_srv_rerror_rate']=form.dst_host_srv_rerror_rate.data

		return redirect(url_for("paquet_prediction"))
	return render_template('analyse_paquet.html',form=form)

@app.route('/importer_fichier', methods=['GET', 'POST'])
def importer_fichier():
    form = FileForm()
    if form.validate_on_submit():
        file = form.file.data
        if file.filename == '':
            flash('Aucun fichier sélectionné')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            session['filename']=filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Fichier téléchargé avec succès')
            return redirect(url_for('resultats_prediction'))
    return render_template('importer_fichier.html', form=form)


@app.route('/resultats_prediction', methods=['GET', 'POST'])
def resultats_prediction():
    
    predictions_protocole = []
    predictions_NoN = []
    predictions_maria = []
    predictions_rose = []
    liste_anormal = []
    predictions_finale=[]
    filename = ""
    maria_classification_report = ""
    rose_classification_report = ""
    repertoire = app.config['UPLOAD_FOLDER']
    # Obtenir la liste des fichiers dans le répertoire
    fichiers = os.listdir(repertoire)

    # Vérifier s'il y a un seul fichier dans le répertoire
    if len(fichiers) == 1:
        # Ouvrir le premier fichier de la liste en mode lecture
        filename = fichiers[0]
        file_path = os.path.join(repertoire, filename)
        print(filename)
        with open(file_path, 'r') as fichier:
            # Charger le fichier CSV en tant que DataFrame
            columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack-type', 'level']
            df = pd.read_csv(file_path, header=None, names=columns)
            df_1_0=pd.read_csv(file_path, header=None, names=columns)
        os.remove(file_path)
        print(df.head())
        dico_protocol = {'tcp': 1, 'udp': 2, 'icmp': 3}
        dico_flag = {'SF': 7, 'S0': 2, 'REJ': 10, 'RSTR': 9, 'SH': 8, 'RSTO': 5, 'S1': 4, 'RSTOS0': 3, 'S3': 6, 'S2': 11, 'OTH': 1}
        dico_service = {'ftp_data': 5, 'other': 30, 'private': 34, 'http': 50, 'remote_job': 3, 'name': 57, 'netbios_ns': 21, 'eco_i': 44, 'mtp': 12, 'telnet': 26, 'finger': 10, 'domain_u': 49, 'supdup': 16, 'uucp_path': 25, 'Z39_50': 23, 'smtp': 39, 'csnet_ns': 36, 'uucp': 19, 'netbios_dgm': 33, 'urp_i': 22, 'auth': 40, 'domain': 52, 'ftp': 14, 'bgp': 58, 'ldap': 46, 'ecr_i': 11, 'gopher': 4, 'vmnet': 64, 'systat': 55, 'http_443': 42, 'efs': 65, 'whois': 1, 'imap4': 38, 'iso_tsap': 32, 'echo': 61, 'klogin': 45, 'link': 24, 'sunrpc': 54, 'login': 43, 'kshell': 13, 'sql_net': 31, 'time': 7, 'hostnames': 59, 'exec': 2, 'ntp_u': 29, 'discard': 28, 'nntp': 9, 'courier': 66, 'ctf': 48, 'ssh': 63, 'daytime': 56, 'shell': 69, 'netstat': 67, 'pop_3': 41, 'nnsp': 51, 'IRC': 17, 'pop_2': 47, 'printer': 27, 'tim_i': 20, 'pm_dump': 35, 'red_i': 70, 'netbios_ssn': 6, 'rje': 18, 'X11': 53, 'urh_i': 8, 'http_8001': 68, 'aol': 62, 'http_2784': 15, 'tftp_u': 37, 'harvest': 60}
        #dico_classement_attaque={'neptune':1, 'warezclient':2, 'ipsweep':0, 'portsweep':0,'teardrop':1, 'nmap':0, 'satan':0, 'smurf':1, 'pod':1, 'back':2,'guess_passwd':2, 'ftp_write':2, 'multihop':2, 'rootkit':2,'buffer_overflow':2, 'imap':2, 'warezmaster':2, 'phf':2, 'land':1,'loadmodule':2, 'spy':2, 'perl':2}
        dico_classement_attaque={'neptune':2,'smurf':2,'apache2':2,'back':1,'processtable':2,'pod':2,'mailbomb':2,'land':2,'teardrop':2,
                          'udpstorm':2,'saint':3,'mscan':3,'satan':3,'nmap':3,'ipsweep':3,'portsweep':3,'guess_passwd':1,'snmpgetattack':1,
                          'snmpguess':1,'ftp_write':1,'imap':1,'phf':1,'warezclient':1,'spy':1,'buffer_overflow':1,'httptunnel':1,'ps':1,'multihop':1,
                          'named':1,'sendmail':1,'loadmodule':1,'xterm':1,'worm':1,'rootkit':1,'xlock':1,'perl':1,'xsnoop':1,'sqlattack':1,'warezmaster':3}
        df['protocol_type'] = df['protocol_type'].map(dico_protocol)
        df['flag'] = df['flag'].map(dico_flag)
        df['service'] = df['service'].map(dico_service)
        
        df.drop(['num_outbound_cmds'], axis=1, inplace=True)

        Maria_dataset(df, model_NoN, model_tcp, model_udp, model_icmp, predictions_maria, predictions_protocole, predictions_NoN,maria_classification_report)

        for i in range(0, len(predictions_maria)):
            if predictions_maria[i] == 1:
                liste_anormal.append(i)

        Rose(df, liste_anormal, model_rose, scaler_rose, predictions_rose)
        j=0
        print("Predictions maria: ",len(predictions_maria))
        print("Liste anormal: ",len(liste_anormal))
        print("Predictions rose sur normal: ",len(predictions_rose))
        
        '''for i in range(0,len(predictions_rose_1_0)):
            if predictions_rose_1_0[i]!=0:
                j+=1
                predictions_maria[liste_normal_1_0[i]]=1
                
        for i in range(0, len(predictions_maria)):
            if predictions_maria[i] == 1:
                liste_anormal.append(i)'''
        
        print("Anormal: ",j)
        print(len(liste_anormal))
        print(len(predictions_maria))
        y_maria=df['attack-type'].apply(normal_or_not).copy()
        y_true_maria=y_maria.values
        y_true_maria=list(int(element) for element in y_true_maria)
        maria_classification_report=classification_report(y_true_maria,predictions_maria)
        NoN_classification_report=classification_report(y_true_maria,predictions_NoN)
        protocole_classification_report=classification_report(y_true_maria,predictions_protocole)

        df_rose=df.iloc[liste_anormal].copy()
        df_rose['attack-type']=df_rose['attack-type'].map(dico_classement_attaque)
        df_rose['attack-type']=df_rose['attack-type'].fillna(1)
        y_true_rose=df_rose['attack-type'].values
        y_true_rose=list(int(element) for element in y_true_rose)
        rose_classification_report=classification_report(y_true_rose,predictions_rose,zero_division=np.NaN)
        

    return render_template('resultats_prediction.html', predictions_protocole=predictions_protocole, predictions_NoN=predictions_NoN, predictions_maria=predictions_maria, predictions_rose=predictions_rose,maria_classification_report=maria_classification_report,rose_classification_report=rose_classification_report, NoN_classification_report=NoN_classification_report,protocole_classification_report=protocole_classification_report,df=df)
    
@app.route('/about')
def about():
    active_page='about'
    return render_template('description.html',active_page=active_page)

@app.route('/architecture')
def architecture():
    active_page='architecture'
    return render_template('architecture.html', active_page=active_page)

@app.route('/contact')
def contact():
    active_page='contact'
    return render_template('contact.html', active_page=active_page)

@app.route('/inscription', methods=['GET','POST'])
def inscription():
    active_page='inscription'
    if request.method=='POST':
        username=request.form['floatingText']
        mail=request.form['floatingInput']
        password=request.form['floatingPassword']
        hashed_password=hash_password(password)
	# Vérification des données et création d'un nouvel utilisateur
        new_user=User(username=username,email=mail,password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('connexion'))
    return render_template('signup.html')

@app.route('/connexion',methods=['GET','POST'])
def connexion():
    active_page='connexion'
    if request.method=='POST':
        mail=request.form['floatingInput']
        password=request.form['floatingPassword']

        # Vérification des informations de l'utilisateur

        #Exemple
        user = User.query.filter_by(email=mail).first()

        if user and verify_password(password,user.password):
            
            login_user(user)
            flash('Vous êtes connecté avec succès!','success')
            session['user_id']=user.id
            session['user_name']=user.username
            session['user_mail']=user.email
            return redirect(url_for('home'))
        else:
            flash('Identifiants invalides. Veuillez réessayer.', 'danger')
    return render_template('signin.html')

@app.route('/deconnexion')
@login_required
def deconnexion():
    active_page='deconnexion'
    logout_user()
    session.pop('user_id',None)
    session.pop('user_name',None)
    session.pop('user_mail',None)
    return redirect(url_for('home'))

@app.route('/reset-db', methods=['POST','GET'])
def reset_db():
    if request.method == 'GET':
        # Code pour réinitialiser la base de données
        db.drop_all()
        db.create_all()
        return "La base de données a été réinitialisée avec succès."
if __name__=='__main__':
	app.run(debug=True)
