//Ross Metcalfe, 2019
//This program is a naive implementation of Multi layer Perceptrons with a sigmoid activation function.
//The neurons are implemented as objects. A better implementation would use matrices to represent network layers
//Left click and Right click adds a training example at the position the mouse was clicked, using X and Y as two predictive attributes
//C clears the dataset and generates a new NeuralNetwork model
//R adds 20 random datapoints
//P pauses the network, when paused new datapoints can still be added, the network won't train on those until P is pressed again to unpause
//The data set is constructed using two attributes, X and Y as predictive attributes, and two goal-field attributes RED and GREEN.
//The goal field attributes are probabilities that the classification is RED or GREEN given the predictive attributes X and Y.

NeuralNetwork testNet; //network testing object
ArrayList<float[]> SVMdata; //positions of record entries, two attributes [X,Y]
ArrayList<float[]> SVMlabels; //classification label of positions, two classification attributes, [RED (0-1), GREEN (0-1)]

//Define and Initialise Layer Sizes of the neural network
int[] layer_sizes = {2,8,2};
//int[] layer_sizes = {2,8,3,2};
//int[] layer_sizes = {2,10,5,2};

//The stepping size global variable used for training the network weights and biases
float stepSize=0.01;


void setup(){
  //size(1000,1000);
  fullScreen();
  
  //Initialise neural network and print values to console
  testNet = new NeuralNetwork(layer_sizes);
  testNet.printNetwork();
  
  //Initialise empty data and label arrays
  SVMdata = new ArrayList<float[]>();
  SVMlabels = new ArrayList<float[]>();
}

void draw(){
  background(255); //white background
  int t0=millis(); //Frame start time
  
  
  //If there is any data to train on and training is not paused,
  if(SVMdata.size()>0&&!paused){
    //repeat training until at least 13 milliseconds has passed to preserve good framerate
    while(millis()-t0<13){
      train(50);
    }
  }
  //testNet.printNetwork();
  noStroke();
  
  //draw the training data and the classification field across the attributes X,Y by dividing the window into 16 pixel sided squares
  drawSVM(16);
  //saveFrame("frames/frame" + nf(frameCount,4) + ".png");
}

float f(float x){
  return sin(x);
}

void drawZero(){
  line(0,0,width,0);
}
void drawF(){
  ArrayList<Float> outputs= new ArrayList<Float>();
  float[] data = new float[1];
  for(float i=0;i<2*PI;i+=0.1){
    data[0] = i;
    outputs.add(f(data[0]));
    if(i!=0){
      line(i-0.1,-outputs.get(outputs.size()-2),i,-outputs.get(outputs.size()-1));
    }
  }
}

void drawSVM(int sqSide){
  
  //Draw the training examples as circles
  int i=0;
  for(float[] data : SVMdata){
    if(SVMlabels.get(i)[0]>SVMlabels.get(i)[1]){
      fill(255,0,0);
    }else{
      fill(0,255,0);
    }
    ellipse((1+data[0])/2*width,(1+data[1])/2*height,10,10);
    i++;
  }
  
  
  //Draw the field of classification
  for(int x=0;x<ceil(1+width/sqSide);x++){
    for(int y=0;y<ceil(1+height/sqSide);y++){
      int posX,posY;
      posX=x*sqSide;
      posY=y*sqSide;
      float[] input=new float[2];
      input[0] = -1+2*float(posX)/width;
      input[1] = -1+2*float(posY)/height;
      float[] classPrediction = testNet.predict(input);
      float classDif = pow(abs(classPrediction[0]-classPrediction[1]),0.3);
      if(classPrediction[0]>classPrediction[1]){
        fill(255*classDif,0,0,80);
      }else{
        fill(0,255*classDif,0,80);
      }
      rect(posX,posY,sqSide,sqSide);
      
    }
  }
}

void train(int times){//Training function, 
  //for a number of rounds (minimum 1),
  for(int i=0;i<max(1,times/ceil(SVMdata.size()));i++){
    //each round trains all examples once
    for(int j=0;j<SVMdata.size();j++){
      float[] data = SVMdata.get(j);
      float[] label = SVMlabels.get(j);
      testNet.learn(data,label);
    }
  }
}

void trainMiniBatch(int times){
  //modified version of the train() function which ensures that each round equal numbers of examples from each classification are trained
  for(int i=0;i<max(1,times/ceil(SVMdata.size()));i++){
    ArrayList<float[]> dataBatchGreen = new ArrayList<float[]>();
    ArrayList<float[]> labelsBatchGreen = new ArrayList<float[]>();
    ArrayList<float[]> dataBatchRed = new ArrayList<float[]>();
    ArrayList<float[]> labelsBatchRed = new ArrayList<float[]>();
    for(int j=0;j<SVMdata.size();j++){
      if(SVMlabels.get(j)[0]==1f){
        dataBatchGreen.add(SVMdata.get(j));
        labelsBatchGreen.add(SVMlabels.get(j));
      }else{
        dataBatchRed.add(SVMdata.get(j));
        labelsBatchRed.add(SVMlabels.get(j));
      }
    }
    while(dataBatchGreen.size()>0 && dataBatchRed.size()>0){
      if(dataBatchGreen.size()>0){
        int j=round(random(0,dataBatchGreen.size()-1));
        float[] data = dataBatchGreen.get(j);
        float[] label = labelsBatchGreen.get(j);
        testNet.learn(data,label);
        dataBatchGreen.remove(j);
        labelsBatchGreen.remove(j);
      }
      if(dataBatchRed.size()>0){
        int j=round(random(0,dataBatchRed.size()-1));
        float[] data = dataBatchRed.get(j);
        float[] label = labelsBatchRed.get(j);
        testNet.learn(data,label);
        dataBatchRed.remove(j);
        labelsBatchRed.remove(j);
      }
    }
  }
}


boolean lMouse = false;
boolean rMouse = false;

boolean paused=false;
void keyPressed(){
  if(key=='p'){
    paused=!paused;
  }
  if(key=='r'){
    for(int i=0;i<20;i++){
      float[] position = new float[2];
      position[0] = random(-1,1);
      position[1] = random(-1,1);
      float[] label = new float[2];
      label[0]=round(random(0,1));
      label[1]=1-label[0];
      SVMdata.add(position);
      SVMlabels.add(label);
    }
  }
  if(key=='c'){
    SVMdata.clear();
    SVMlabels.clear();
    testNet = new NeuralNetwork(layer_sizes);
  }
}
void mousePressed(){
  if(mouseButton==LEFT){
    if(!lMouse){
      lMouse=true;
      float[] position = new float[2];
      position[0] =-1+2*float(mouseX)/width;
      position[1] = -1+2*float(mouseY)/height;
      float[] label = new float[2];
      label[0]=1f;
      label[1]=0f;
      SVMdata.add(position);
      SVMlabels.add(label);
    }
  }else if(mouseButton==RIGHT){
    if(!rMouse){
      rMouse=true;
      float[] position = new float[2];
      position[0] = -1+2*float(mouseX)/width;
      position[1] = -1+2*float(mouseY)/height;
      float[] label = new float[2];
      label[0]=0f;
      label[1]=1f;
      SVMdata.add(position);
      SVMlabels.add(label);
    }
  }
}

void mouseReleased(){
  if(mouseButton==LEFT){
    if(lMouse){
      lMouse=false;
    }
  }else if(mouseButton==RIGHT){
    if(rMouse){
      rMouse=false;
    }
  }
}