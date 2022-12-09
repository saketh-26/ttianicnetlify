from Titanic_Model import survive as sp
import numpy as np
from flask import Flask,request,render_template
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        form = request.form
        age = float(form['age'])
        sibsp = int(form['sibsp'])
        parch = int(form['parch'])
        fare = float(form['fare'])
        gender = form['gender']
        pclass = int(form['pclass'])
        place = form['place']

        p = []
        p +=[age,sibsp,parch,fare]
        if gender == "M":
            p+=[1]
        else:
            p+=[0]
        if pclass == 2:
            p+=[1,0]
        elif pclass == 3:
            p+=[0,1]
        else:
            p+=[0,0]
        if place.lower() == "queenstown":
            p+=[1,0]
        elif place.lower() == "southampton":
            p+=[0,1]
        else:
            p+=[0,0]
        arr = np.array([p])
        res = sp(arr)
        result = "Survived" if res[0] == 1 else "Not Survived"

        return render_template('index.html',res=result)

    return render_template('index.html',res = "Fill the details and Click Submit")
if __name__ == "__main__":
    app.run()

            
    
