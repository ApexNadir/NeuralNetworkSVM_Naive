//Ross Metcalfe, 2019



class NeuralNetwork{
  Neuron[][] network;
  NeuralNetwork(int[] layer_sizes){
    network = new Neuron[layer_sizes.length][];
    for(int i=0;i<layer_sizes.length;i++){
      network[i] = new Neuron[layer_sizes[i]];
      for(int j=0;j<layer_sizes[i];j++){
        if(i==0){
          network[i][j] = new Neuron(null,true);
        }else if(i<layer_sizes.length-1){
          network[i][j] = new Neuron(network[i-1],true);
        }else{
          network[i][j] = new Neuron(network[i-1],false);
        }
      }
    }
  }
  
  float[] predict(float[] input){
    for(int j=0;j<network[0].length;j++){
      network[0][j].value = input[j];
    }
    for(int i=1;i<network.length;i++){
      for(int j=0;j<network[i].length;j++){
        network[i][j].update();
      }
    }
    float[] output=new float[network[network.length-1].length];
    for(int j=0;j<network[network.length-1].length;j++){
      output[j] = network[network.length-1][j].value;
    }
    
    return output;
  }
  
  void learn(float[] input, float[] output){
    float[] prediction = predict(input);
    float[] gradient = new float[prediction.length];
    for(int i=0;i<gradient.length;i++){
      gradient[i] = output[i]-prediction[i];
      //println(gradient[i]);
      network[network.length-1][i].addGradient(gradient[i]);
    }
    
    for(int i=network.length-1;i>0;i--){
      for(int j=0;j<network[i].length;j++){
        network[i][j].backprop();
      }
    }
    
    
  }
  
  void printNetwork(){
    for(int i=0;i<network.length;i++){
      for(int j=0;j<network[i].length;j++){
        Neuron n = network[i][j];
        if(n.inputs==null){
          print("[" + n.value + "]");
        }else{
          for(int input=0;input<n.inputs.length;input++){
            print(n.weights[input]+ ", ");
          }
          print(" + " + n.bias);
          print(" = [" + network[i][j].value + "]");
        }
      }
      print("\n");
    }
  }
  
}

class Neuron{
  float[] weights;
  float bias;
  Neuron[] inputs;
  float incomingGradient;
  float value;
  boolean activation;
  Neuron(Neuron[] inputs0, boolean activationFunction){
    activation = activationFunction;
    if(inputs0==null){
      return;
    }
    inputs = inputs0;
    weights = new float[inputs.length];
    for(int i=0;i<inputs.length;i++){
      weights[i] = randomGaussian()/pow(inputs.length,0.5);
    }
    bias = 0;
  }
  
  void update(){
    if(inputs==null){return;}
    incomingGradient=0;
    value=0;
    for(int i=0;i<inputs.length;i++){
      value+=inputs[i].value*weights[i];
    }
    value+=bias;
    if(activation){
      value=activation(value);
    }
  }
  
  void addGradient(float incAdd){
    incomingGradient+=incAdd;
  }
  
  void backprop(){
    float valueGrad;
    if(activation){
      valueGrad=sigmoidGradientAlready()*incomingGradient;
    }else{
      valueGrad=incomingGradient;
    }
    for(int i=0;i<inputs.length;i++){
      inputs[i].addGradient(valueGrad*weights[i]);
      weights[i]+=valueGrad*inputs[i].value*stepSize;
    }
    bias+=valueGrad*stepSize;
  }
  
  float sigmoidGradientAlready(){
    return value*(1-value);
  }
  
  
}


static float sigmoidGradient(float x){
  float sigx=activation(x);
  return sigx*(1-sigx);
}

static float activation(float x){
  return 1/(1+exp(-x));
}