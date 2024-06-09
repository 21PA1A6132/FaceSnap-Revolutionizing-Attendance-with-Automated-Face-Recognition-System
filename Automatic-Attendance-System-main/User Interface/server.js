const { initializeApp, cert } = require("firebase-admin/app");
const { getFirestore } = require("firebase-admin/firestore");
var hash = require("password-hash");
const multer = require('multer');
const path = require('path');
const bodyparser = require("body-parser");
const { spawn } = require('child_process');
const fs = require('fs');


var serviceAccount = require("./key.json");

initializeApp({
  credential: cert(serviceAccount),
});

const db = getFirestore();

const express = require("express");

const app = express();

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, './uploads'); // Store uploaded files in the 'uploads' directory
  },
  filename: function (req, file, cb) {
    cb(null, 'temp.mp4'); // Rename uploaded file to 'temp.mp4'
  }
});

const upload = multer({ storage: storage });

app.use(bodyparser.urlencoded({ extended: true }));
app.use(bodyparser.json());

app.use(express.json());

app.use(express.static("public"));

app.set('view engine', 'ejs');

app.get("/", (req, res) => {
    res.sendFile(__dirname + "/StudentLogin.html");
  });

  app.get("/FacultyLogin.html", (req, res) => {
    res.sendFile(__dirname + "/Facultylogin.html");
  });

  app.get("/StudentLogin.html", (req, res) => {
    res.sendFile(__dirname + "/StudentLogin.html");
  });
  
  app.get("/logout", (req, res) => {
    res.sendFile(__dirname + "/StudentLogin.html");
  });

  
  app.post("/studentloginsubmit", (req, res) => {
    db.collection("StudentLogin")
      .where("username", "==", req.body.inuser)
      .get()
      .then((docs) => {
        if (docs._size == 0) {
          res.sendFile(__dirname + "/logininvalid.html");
        } else {
          docs.forEach((doc) => {
            if (hash.verify(req.body.inpass, doc.data().password)) {
              
            } else {
              res.sendFile(__dirname + "/logininvalid.html");
            }
          });
        }
      });
  });

  app.post("/FacultyLoginSubmit", (req, res) => {
    db.collection("LecturerLogin")
      .where("username", "==", req.body.username)
      .get()
      .then((docs) => {
        if (docs._size == 0) {
            res.send("<script>window.location.href = 'Studentlogin.html'; window.alert(' Invalid Credentials ');</script>");
        } else {
          docs.forEach((doc) => {
            if (hash.verify(req.body.password, doc.data().password)) {
                res.sendFile(__dirname + "/FacultyPage.html")
            } else {
                res.send("<script>window.location.href = 'Studentlogin.html'; window.alert(' Invalid Credentials ');</script>");
            }
          });
        }
      });
  });


  app.get("/class_details", (req, res) => {
    res.render('FacultyPage1.ejs', { year: req.query.year , branch : req.query.branch , section : req.query.section , period : req.query.period , date : req.query.date });
  });

  app.post('/upload', upload.single('video'), (req, res) => {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }
  
    // Execute the Python script
    const pythonProcess = spawn('python', ['web_predict1.py']); // Replace 'your_script.py' with the actual name of your Python script
  
    pythonProcess.stdout.on('data', (data) => {
      console.log(`stdout: ${data}`);
    });
  
    pythonProcess.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
    });
  
    pythonProcess.on('close', (code) => {
      console.log(`child process exited with code ${code}`);

      fs.readFile('data.json', 'utf8', (err, data) => {
        if (err) {
          console.error('Error reading file:', err);
          res.status(500).send('Error reading file');
          return;
        }
    
        try {
          // Parse the JSON data into a JavaScript object
          const jsonData = JSON.parse(data);
          const Newdata = JSON.parse(JSON.stringify(jsonData));
          jsonData.date = req.body.date;
          jsonData.year = req.body.year;
          jsonData.branch = req.body.branch;
          jsonData.section = req.body.section;
          jsonData.period = req.body.period;

          console.log(jsonData);
    
          db.collection('attendance').add(jsonData);
          
          temp = '<!DOCTYPE html> <html> <head> <title>Student Page</title> <style> body { margin: 0; padding: 0; } #myVideo { position: fixed; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: -1; } .container { position: relative; z-index: 1; } .content { color: azure; text-align: center; width: 45%; padding: 12px; margin-top: 8%; margin-bottom: 16px; margin-left: 25%; align-items: center; box-shadow: 0 0 15px rgba(236, 231, 231, 0.877); border-radius: 10px; box-sizing: border-box; backdrop-filter: blur(3px); transition: transform 0.3s ease; } .content:hover { backdrop-filter: blur(9px); transform: scale(1.05); box-shadow: 0 0 15px rgb(228, 200, 231); } a { padding-left: 8px; padding-right: 8px; padding-top: 8px; padding-bottom: 8px; height: 50px; width: 110px; border-radius: 5px; background-color: aliceblue; text-decoration: none; } a:hover{ background-color: darkgray; } table { margin-top: 3%; margin-left: 30%; width: 40%; } h1 { font-family: monospace; } </style> </head> <body> <video autoplay muted loop id="myVideo"> <source src="http://localhost:3000/black.mp4" type="video/mp4"> </video> <div class="container"> <div class="content"> <h1>Check Your Attendance</h1> <p><b>Date:</b>' + jsonData.date + ' </p> <p><b>Year:</b> ' + jsonData.year + '</p> <p><b>Branch:</b> ' + jsonData.branch + '</p> <p><b>Section:</b>' + jsonData.section + ' </p> <p><b>Period:</b>' + jsonData.period + ' </p> <table border="1"> <tr> <td><b>Register</b></td> <td><b>Status(P/A)</b></td> </tr> <tbody>';
          for(let key in Newdata){
            if(key!=="date" || key!=="year" ||key!=="branch" ||key!=="section" ||key!=="period")
            {
              if(Newdata[key]==-1)
              {
                temp+='<tr><td>'+ key +'</td><td>P</td></tr>';
              }
              else{
                temp+='<tr><td>'+ key +'</td><td>A</td></tr>';
              }
              
            }
          }
          temp=temp+'</tbody> </table> </div> </div> </body> </html>';

          res.send(temp);


        } catch (error) {
          console.error('Error parsing JSON data:', error);
          res.status(500).send('Error parsing JSON data');
        }
      });

      
    });
  }); 

  app.get("/view", (req, res) => {
    res.sendFile(__dirname + "/view.html");
  });

  app.get("/view.html", (req, res) => {
    res.sendFile(__dirname + "/view.html");
  });

  app.get("/view_submit", (req, res) => {
    db.collection('attendance')
    .where('date' , "==",req.query.date)
    .where('year' , "==",req.query.year)
    .where('branch' , "==",req.query.branch)
    .where('section' , "==",req.query.section)
    .where('period' , "==",req.query.period)
    .get().then((snapshot) => {
      
    snapshot.forEach((doc)=> {
      
      var jsonData = doc.data();
      temp = '<!DOCTYPE html> <html> <head> <title>Student Page</title> <style> body { margin: 0; padding: 0; } #myVideo { position: fixed; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: -1; } .container { position: relative; z-index: 1; } .content { color: azure; text-align: center; width: 45%; padding: 12px; margin-top: 8%; margin-bottom: 16px; margin-left: 25%; align-items: center; box-shadow: 0 0 15px rgba(236, 231, 231, 0.877); border-radius: 10px; box-sizing: border-box; backdrop-filter: blur(3px); transition: transform 0.3s ease; } .content:hover { backdrop-filter: blur(9px); transform: scale(1.05); box-shadow: 0 0 15px rgb(228, 200, 231); } a { padding-left: 8px; padding-right: 8px; padding-top: 8px; padding-bottom: 8px; height: 50px; width: 110px; border-radius: 5px; background-color: aliceblue; text-decoration: none; } a:hover{ background-color: darkgray; } table { margin-top: 3%; margin-left: 30%; width: 40%; } h1 { font-family: monospace; } </style> </head> <body> <video autoplay muted loop id="myVideo"> <source src="http://localhost:3000/black.mp4" type="video/mp4"> </video> <div class="container"> <div class="content"> <h1>Check Your Attendance</h1> <p><b>Date:</b>' + jsonData.date + ' </p> <p><b>Year:</b> ' + jsonData.year + '</p> <p><b>Branch:</b> ' + jsonData.branch + '</p> <p><b>Section:</b>' + jsonData.section + ' </p> <p><b>Period:</b>' + jsonData.period + ' </p> <table border="1"> <tr> <td><b>Register</b></td> <td><b>Status(P/A)</b></td> </tr> <tbody>';
      for(let key in jsonData){
        if(key !== "date" && key !== "year" && key !== "branch" && key !== "section" && key !== "period")
        {
          if(jsonData[key]==-1)
          {
            temp+='<tr><td>'+ key +'</td><td>P</td></tr>';
          }
          else{
            temp+='<tr><td>'+ key +'</td><td>A</td></tr>';
          }
          
        }
      }
      temp=temp+'</tbody> </table> </div> </div> </body> </html>';

      res.send(temp);
      
    })
  });
  });

app.listen(3000);
