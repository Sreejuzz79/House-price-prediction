# House Price Prediction with PyTorch

A simple neural network implementation that predicts house prices based on house size using PyTorch. This project demonstrates the fundamentals of linear regression with deep learning.

##  Project Overview

This project creates a basic machine learning model that learns the relationship between house size and price. Think of it like teaching a computer to be a real estate appraiser - it looks at the size of houses and learns to estimate their prices based on patterns in the data.

##  What It Does

- **Generates synthetic data**: Creates 100 sample houses with random sizes and corresponding prices
- **Builds a neural network**: Uses a simple linear layer to learn the size-to-price relationship  
- **Trains the model**: Optimizes the network to minimize prediction errors
- **Tracks progress**: Monitors how well the model learns over time

##  Model Architecture

The neural network is beautifully simple:
- **Input**: House size (1 feature)
- **Hidden Layer**: Single linear transformation 
- **Output**: Predicted price (1 value)

It's like having a mathematical function that transforms "square footage" into "dollar amount" - and the network learns the best way to do this transformation.

##  Data Generation

The synthetic dataset follows this relationship:
```
house_price = house_size × 2 + random_noise + 100
```

- **House sizes**: Normally distributed around 10 units (±5 variation)
- **Price formula**: Each unit of size adds ~$2 to the base price of $100
- **Noise**: Random variations make it realistic (real estate isn't perfectly predictable!)

##  Getting Started

### Prerequisites
```bash
pip install torch matplotlib
```

### Running the Code
```bash
python house_price_prediction.py
```

### Expected Output
You'll see training progress every 2 epochs:
```
epoch : 0 loss : 2845.234
epoch : 2 loss : 1203.567
epoch : 4 loss : 567.123
...
```

The loss should decrease over time, showing the model is learning!

##  Code Structure

- **Data Generation**: Creates realistic house size and price data
- **Model Definition**: Simple linear neural network class
- **Training Loop**: Uses Adam optimizer with MSE loss
- **Progress Tracking**: Prints loss every 2 epochs

##  How It Works

1. **Forward Pass**: Model makes price predictions based on house sizes
2. **Loss Calculation**: Compares predictions to actual prices using Mean Squared Error
3. **Backward Pass**: Calculates gradients (how to improve)
4. **Parameter Update**: Adjusts model weights to reduce error
5. **Repeat**: Does this 100 times to gradually improve

Think of it like learning to throw darts - each throw (epoch) teaches you how to adjust your aim for the next one!

##  Learning Outcomes

This project demonstrates:
- Basic PyTorch neural network creation
- Synthetic data generation
- Training loop implementation
- Loss function usage (MSE)
- Gradient-based optimization (Adam)

##  Future Enhancements

- Add data visualization with matplotlib
- Include model evaluation metrics
- Save and load trained models
- Add validation dataset
- Experiment with different architectures
- Use real estate datasets

##  Notes

- The model uses synthetic data, so results are predictable
- Adam optimizer with learning rate 0.01 works well for this simple problem
- MSE loss is perfect for regression tasks like price prediction

##  Contributing

Feel free to fork this project and experiment with:
- Different data distributions
- Various neural network architectures  
- Alternative optimizers and loss functions
- Real-world datasets

---

*This project serves as a great introduction to PyTorch and neural network fundamentals. Happy learning!* 🚀
