import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np


def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Calculate the number of samples in the test set
    test_samples = int(len(X) * test_size)

    # Split the indices into train and test sets
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    # Split the data based on indices
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test



filepath = r".\default of credit card clients.xls"
data0 = pd.read_excel(filepath, index_col=False)#换成自己电脑中文件保存的路径
df=pd.DataFrame(data0)
x=df.iloc[:,1:-1]#取出特征数据
labels=df.iloc[:,-1]#取出标签数据


#分类变量处理
Gender=df.iloc[:,2]#取出性别数据
Marriage=df.iloc[:,4]#取出婚姻数据
encode = pd.get_dummies(Gender)#创建哑变量
encode1 = pd.get_dummies(Marriage)
newdata = pd.concat([x,encode,encode1],axis=1)#将原始特征、哑变量的数据合并
newdata.head()
newdata.drop(["SEX","MARRIAGE"],axis=1,inplace=True)#删除原来的性别、婚姻分类变量
newdata.columns = ["LIMIT_BAL",'EDUCATION','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','MALE','FEMALE','MARRIED','DATING','SINGLE','OTHERS']
xList=newdata.iloc[:,:]

numerical_columns = xList.select_dtypes(include=['float64', 'float32', 'int64', 'int32', 'int16', 'int8', 'uint8'])

x_tensor = torch.tensor(numerical_columns.values, dtype=torch.float32)

mean = x_tensor.mean(dim=0)
std = x_tensor.std(dim=0)
x_scaled = (x_tensor - mean) / std


X_train, X_test, y_train, y_test = custom_train_test_split(x_scaled, labels, test_size=0.2, random_state=42)
# Define the dimensions of input, hidden, and output layers
print(X_train[:6])
print(y_train[:6])

# Define the dimensions of input, hidden, and output layers
input_size = X_train.shape[1]  # Number of input features
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 1  # Number of output classes (binary classification)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create the model instance
model = MLP(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Convert training data to PyTorch tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Convert test data to PyTorch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Initialize an empty list to store all predictions
all_predictions = []

# Make predictions for each sample in the test set
for i in range(len(X_test)):
    # Convert sample to PyTorch tensor
    sample_tensor = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        prediction = model(sample_tensor)

    # Convert prediction to binary (0 or 1) using sigmoid activation function
    prediction_binary = 1 if prediction > 0.5 else 0

    # Append prediction to the list
    all_predictions.append(prediction_binary)

# Convert list of predictions to numpy array
all_predictions = np.array(all_predictions)
fc1_weights = model.fc1.weight
fc1_bias = model.fc1.bias
fc2_weights = model.fc2.weight
fc2_bias = model.fc2.bias


# Print some predictions
print("Sample predictions:")
for i in range(len(X_test)):  # Print predictions for all samples
    print(f"Sample {i + 1} - Predicted Class: {all_predictions[i]}")



