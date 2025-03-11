#1-Greast common dividor
def gcd(a,b):
    if a==0: 
        return b
    return gcd(b%a,a)

#2-prime number
def is_prime(n):
    if (n<=1):
        return False
    for i in range(2,n):
        if(n%i==0):
            return False
    return True

#3-ugly number
def isugly(num):
    if num==0: return False
    factor=[2,3,5]
    for f in factor:
        while num%f==0:
            num/=f
    return num==1
#4-sum of first natrual numbers
def findsum(n):
    return n*(n+1)/2

#5- celsius to fehrenheit
def celsius_to_fahrenheit(cel):
    return (cel*9/5)+32

#6-count digits in number
def count_digits_string(num):
    if num==0: return 1
    return len(str(abs(num)))
#7- palindrome number
def is_palindrome(x):
    if x<0:
        return False
    s=str(x)
    return s==s[::-1]
#8- factroila of number
def fact(n):
    if n==0:
        return 1
    return n*fact(n-1)
#9-digits in factroial
import math
def digitsinfactroial(n):
    if n<0: return 0
    if n<=1: return 1
    result=0
    for i in range(n,n+1):
        result+=math.log10(i)
    return int(result)+1

#10- trailing zero
def find_trailing_zeros(n):
    if n<0: return -1
    count=0
    i=5
    while n//i>=1:
        count +=n//i
        i*=5
    return count
#11-least common multipler
def lcm(a,b):
    res=max(a,b)
    while True:
        if res%a==0 and res%b==0:
            return res
        res+=1
    return res
#12-print divisors
def print_divisors(n):
	i = 1
	while i <= n :
		if (n % i==0) :
			print (i,end=" ")
		i = i + 1
		
#13-pow of x
def power(x, n):
    # Initialize result by 1
    pow = 1
    for _ in range(n):
        pow *= x
    return pow
#14- reverse integer
def reverse(x):
    reversed_x=0
    sign=1 if x>=0 else -1
    x=abs(x)
    while x>0:
        digit=x%10
        reversed_x=reversed_x*10+digit
        x//=10
    
    if reversed_x<-2**31 or reversed_x>2**31-1:
        return 0
    