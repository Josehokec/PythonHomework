"""
factorial_function
writen by lsz
"""

def factorial_function(n):
    """
    Factorial function
    argus: integer n and its value more than 0
    return: 
       if n is not integer return -1
       if n less than 0 return 0
       if n equals 0 return 1
       else return its factorial  
    """
    if isinstance(n, int) != True:
        print("argument error: the argument should be a integer")
        return -1
    elif n < 0:
        print("argument error: the argument should be positive")
        return 0
    elif n == 0:
        return 1;
    else:
        ans = 1
        for i in range(1, n + 1):
            ans = ans * i
        return ans    

if __name__ == '__main__':
    print("question1 : factorial function")
    print(">test sample : 5\nthe result is: ", factorial_function(5))
    print(">test sample : 12\nthe result is: ", factorial_function(12))    
    print(">test sample : 0\nthe result is: ", factorial_function(0))    
    print(">test sample : -1\nthe result is: ", factorial_function(-1))    
    print(">test sample : Hello world\nthe result is: ", factorial_function("Hello world"))
    
