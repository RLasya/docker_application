import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt

# Set the title of the app
st.title("Mathematical Applications")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose a mathematical application",
    ("Basic Arithmetic", "Exponentiation & Logarithms", "Trigonometric Functions", "Square Root & Factorial", "Expression Evaluation", "Plotting")
)

# Basic Arithmetic
if option == "Basic Arithmetic":
    st.header("Basic Arithmetic Operations")
    num1 = st.number_input("Enter the first number", value=0.0)
    num2 = st.number_input("Enter the second number", value=0.0)
    
    operation = st.selectbox("Select operation", ("Addition", "Subtraction", "Multiplication", "Division"))
    
    if operation == "Addition":
        result = num1 + num2
    elif operation == "Subtraction":
        result = num1 - num2
    elif operation == "Multiplication":
        result = num1 * num2
    elif operation == "Division":
        if num2 != 0:
            result = num1 / num2
        else:
            result = "Cannot divide by zero"
    
    st.write(f"Result: {result}")

# Exponentiation & Logarithms
elif option == "Exponentiation & Logarithms":
    st.header("Exponentiation & Logarithms")
    num = st.number_input("Enter a number", value=0.0)
    
    exponent = st.number_input("Enter the exponent", value=1.0)
    result_exp = num ** exponent
    st.write(f"{num} raised to the power of {exponent} is {result_exp}")
    
    if num > 0:  # Logarithms are only defined for positive numbers
        base = st.selectbox("Choose logarithm base", ("Natural Log (ln)", "Base 10 Log"))
        if base == "Natural Log (ln)":
            result_log = math.log(num)
        elif base == "Base 10 Log":
            result_log = math.log10(num)
        st.write(f"Logarithm of {num} is {result_log}")
    else:
        st.write("Logarithms are undefined for non-positive numbers")

# Trigonometric Functions
elif option == "Trigonometric Functions":
    st.header("Trigonometric Functions")
    angle_deg = st.number_input("Enter angle in degrees", value=0.0)
    angle_rad = math.radians(angle_deg)  # Convert to radians
    
    sine = math.sin(angle_rad)
    cosine = math.cos(angle_rad)
    tangent = math.tan(angle_rad)
    
    st.write(f"Sine of {angle_deg}° is {sine}")
    st.write(f"Cosine of {angle_deg}° is {cosine}")
    st.write(f"Tangent of {angle_deg}° is {tangent}")

# Square Root & Factorial
elif option == "Square Root & Factorial":
    st.header("Square Root & Factorial")
    num = st.number_input("Enter a number", value=0.0)
    
    if num >= 0:
        square_root = math.sqrt(num)
        st.write(f"Square root of {num} is {square_root}")
    else:
        st.write("Square root is not defined for negative numbers")
    
    if num.is_integer() and num >= 0:
        factorial = math.factorial(int(num))
        st.write(f"Factorial of {int(num)} is {factorial}")
    else:
        st.write("Factorial is only defined for non-negative integers")

# Expression Evaluation
elif option == "Expression Evaluation":
    st.header("Evaluate a Mathematical Expression")
    expression = st.text_input("Enter a mathematical expression", "")
    
    if expression:
        try:
            result = eval(expression)
            st.write(f"Result: {result}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Plotting
elif option == "Plotting":
    st.header("Plotting Mathematical Functions")
    
    # Select a function to plot
    function_type = st.selectbox("Select function to plot", ("y = x^2", "y = sin(x)", "y = cos(x)", "y = tan(x)"))
    
    # Generate x values
    x_values = np.linspace(-10, 10, 400)
    
    if function_type == "y = x^2":
        y_values = x_values ** 2
        title = "y = x^2"
    elif function_type == "y = sin(x)":
        y_values = np.sin(x_values)
        title = "y = sin(x)"
    elif function_type == "y = cos(x)":
        y_values = np.cos(x_values)
        title = "y = cos(x)"
    elif function_type == "y = tan(x)":
        y_values = np.tan(x_values)
        title = "y = tan(x)"
        y_values = np.clip(y_values, -10, 10)  # To prevent extreme values for tan(x)
    
    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label=title)
    plt.title(f"Plot of {title}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="best")
    st.pyplot(plt)
