# Neural network that takes in dataset with three binary values. It should return the first and ignore the other two
UserEpochs = int(input("How many epochs would you like to train the neural network for? ")) + 1
UserDataSize = int(input("How many examples would you like to use to train the neural network? "))

# Generate dataset
import numpy as np
import matplotlib.pyplot as plt

DataSet_Input = np.random.randint(2, size= (UserDataSize,3))
DataSet_Output = DataSet_Input[:,[0]]

# Create Sigmoid function and its derivative to be applied at each perceptron
def Sigmoid(x):
    return 1/(1+np.exp(-x))

def SigmoidDerivative(x):
    return (np.exp(x))/(1+np.exp(x))**2

#Create 3 by 1 matrix representing perceptron weights
weights = 2*np.random.random((3,1))-1
weights0 = str(weights)

#Forward propigation
def Return_Output(Input):
    Input = Input.astype(float)
    Output = Sigmoid(np.dot(Input,weights))
    return Output

#Backward propigation
def Training(TrainInputData, TrainOutputData, weights, Epochs):
    weightlist = []
    AdjustList = []
    for epoch in range(Epochs):
        weightlist.append((float(weights[0]), float(weights[1]), float(weights[2])))
        Output = Return_Output(TrainInputData)
        Error = TrainOutputData - Output
        Adjust = np.dot(TrainInputData.T, Error * SigmoidDerivative(TrainOutputData))
        AdjustList.append(Adjust)
        weights += Adjust
    return list((weightlist, AdjustList))

#Generate graph showing weights over epochs
TrainingOutput = Training(DataSet_Input,DataSet_Output, weights,UserEpochs)
WeightsListComb = TrainingOutput[0]
AdjustListComb = TrainingOutput[1]


WeightList1 = []
WeightList2 = []
WeightList3 = []
for weight in WeightsListComb:
    WeightList1.append(weight[0])
    WeightList2.append(weight[1])
    WeightList3.append(weight[2])

plt.title("Weights vs Epoch")
plt.plot(WeightList1, label = "First Neuron")
plt.plot(WeightList2, label = "Second Neuron")
plt.plot(WeightList3, label = "Third Neuron")
plt.ylabel('Sigmoid Neuron Weight')
plt.xlabel("Epochs")
plt.legend()
plt.show()

AdjustList1 = []
AdjustList2 = []
AdjustList3 = []

for adjust in AdjustListComb:
    AdjustList1.append(adjust[0])
    AdjustList2.append(adjust[1])
    AdjustList3.append(adjust[2])

plt.title(" Weight Adjustments vs Epoch")
plt.plot(AdjustList1, label = "First Neuron")
plt.plot(AdjustList2, label = "Second Neuron")
plt.plot(AdjustList3, label = "Third Neuron")
plt.ylabel('Sigmoid Neuron Weight Adjustment')
plt.xlabel("Epochs")
plt.legend()
plt.show()

def PrintResults():
    User_input = np.array([1,0,1])
    Output = Return_Output(User_input)

    print("The training arrays are:")
    print(DataSet_Input)
    print(DataSet_Output)
    print("\n" + "The initial randomly generated weights are:")
    print(weights0)
    print("\n" + "The new weights after " + str(UserEpochs) + " epochs are: ")
    print(weights)
    print("\n" + "The result for the user input array " + str(User_input) + " is:")
    print(Output)
    print("≈" + str(round(float(Output))))

PrintResults()

