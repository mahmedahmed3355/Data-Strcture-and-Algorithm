#1-check whether the string is Symmetrical or Palindrome
#check if symetric
half=len(s)//2
sym=s[:half]==s[half:] if len(s)%2==0 else s[:half]==s[half+1:]
#check if palindrome
S=S[::-1]
#2-reverse words in given string
reverse_word="".join(s.split()[::-1])

#3-capitalize the first and last character of each word in a string
s = "welcome to geeksforgeeks"
res = ' '.join(
    map(
        lambda word: word[0].upper() + word[1:-1] + word[-1].upper()  # Capitalize first and last character
        if len(word) > 1 else word.upper(),  # If word length is 1, capitalize the whole word
        s.split()  # Split `s`into words
    )
)
print(res)
#4-check if a string has at least one letter and one number
l=any(c.isalpha()for c in s)
n=any(c.isdigit()for c in s)
if I and n : print("True") else: print("False")

#5- Accept the Strings Which Contains all Vowels
v="aeiou"
if all(i in s.lower()for i in v):
    print(True)
else:
    print("False")

#6-Count the Number of matching characters in a pair of string
s1 = "VISHAKSHI"
s2 = "VANSHIKA"
res=len(set(s1.lower()).intersection(set(s2.lower())))
#another solution
from collections import Counter
c1=Counter(s1.lower())
c2=Counter(s2.lower())
res=sum((c1&c2).values())
#7-Remove All Duplicates from a Given String in Python
from collections import OrderedDict
s="geeksforgeeks"
res="".join(OrderedDict.formkeys(s))
#8-Maximum Frequency Character in String 
freq=Counter(s)
max_char=max(freq,key=freq.get)
print(max_char)
#9-Odd Frequency Characters 
freq=Counter(s)
res=[ch for ch,count in freq.items() if count%2!=0]
print(res)
#10-Find words which are greater than given length k
def string(str,k):
    string=[]
    text=str.split("")
    for x in text:
        if len(x)>k:
            string.append(x)
    return string
#11-Find all close matches of input string from a list
strr = ["Lion", "Li", "Tiger", "Tig"]
a = "Lion"
for s in strr:
    if s.startwith(a) or a.startwith(strr):
        print(s)
#12-Find Uncommon Words from Two Strings
s1 = "Geeks for Geeks"
s2 = "Learning from Geeks for Geeks"

# count word occurrences
count = Counter(s1.split()) + Counter(s2.split())

# extract uncommon words
res = [word for word in count if count[word] == 1]

print(res)
#13-Permutation of a given string using inbuilt function
import itertools

s = "GFG"
li = [''.join(p) for p in itertools.permutations(s)]
print(li)
#14-Print the middle character of a string
def printMiddleCharacter(str):
     
    # Finding string length
    length = len(str);
 
    # Finding middle index of string
    middle = length // 2;
 
    # Print the middle character
    # of the string
    print(str[middle]);
#15-Generate Valid IP Addresses from String

# Function to check whether segment is valid or not.
def isValid(s):
    n = len(s)

    # Segment of length one is always valid
    if n == 1:
        return True

    # Converting string into integer
    val = int(s)

    # Invalid case: If it has a preceding zero or 
    # its value is greater than 255
    if s[0] == '0' or val > 255:
        return False

    return True

# Recursive helper function to generate valid IP address
def generateIpRec(s, index, curr, cnt, res):
    temp = ""

    # Base case: Reached end of string and 
    # all 4 segments were not completed
    if index >= len(s):
        return

    if cnt == 3:
        temp = s[index:]

        # Checking 4th (last) segment of IP address
        if len(temp) <= 3 and isValid(temp):
            res.append(curr + temp)

        return

    for i in range(index, min(index + 3, len(s))):
        # Creating next segment of IP address
        temp += s[i]

        # If the created segment is valid
        if isValid(temp):
            # Generate the remaining segments of IP
            generateIpRec(s, i + 1, curr + temp + ".", cnt + 1, res)

# Function to generate valid IP address
def generateIp(s):
    res = []
    generateIpRec(s, 0, "", 0, res)
    return res

if __name__ == "__main__":
    s = "255678166"
    res = generateIp(s)
    
    for ip in res:
        print(ip)
#16-Convert numeric words to numbers

from word2number import w2n
 
# initializing string
test_str = "zero four zero one"
 
# printing original string
print("The original string is : " + test_str)
 
# Convert numeric words to numbers
# Using word2number
res = w2n.word_to_num(test_str)
 
# printing result
print("The string after performing replace : " + str(res))

#17-Consecutive characters frequency
from itertools import groupby

# Input string
s = "aaabbccaaaa"

# Group and count consecutive characters
res = [''.join(g) for k, g in groupby(s)]

print(res) 
#18-String slicing in Python to Rotate a String
def rotate(input,d):
 
    # slice string in two parts for left and right
    Lfirst = input[0 : d]
    Lsecond = input[d :]
    Rfirst = input[0 : len(input)-d]
    Rsecond = input[len(input)-d : ]
 
    # now concatenate two parts together
    print ("Left Rotation : ", (Lsecond + Lfirst) )
    print ("Right Rotation : ", (Rsecond + Rfirst))

#17-find minimum number of rotations to obtain actual string
def min_rotations(s1, s2):
    """
    Finds the minimum number of string rotations for s1 to obtain s2.

    Args:
        s1: The string to rotate.
        s2: The target string.

    Returns:
        The minimum number of rotations, or -1 if s2 cannot be obtained from s1.
    """
    if len(s1) != len(s2):
        return -1

    if s1 == s2:
        return 0

    for rotations in range(1, len(s1) + 1):
        rotated_s1 = s1[rotations:] + s1[:rotations]
        if rotated_s1 == s2:
            return rotations

    return -1
#18-Check if String Contains Substring 
# Take input from users
MyString1 = "A geek in need is a geek indeed"

if "need" in MyString1:
    print("Yes! it is present in the string")
else:
    print("No! it is not present")

#19-All substrings Frequency in String
# initializing string
test_str = "abababa"
 
# printing original string
print("The original string is : " + str(test_str))
 
# list comprehension to extract substrings
temp = [test_str[idx: j] for idx in range(len(test_str)) for j in range(idx + 1, len(test_str) + 1)]
 
# loop to extract final result of frequencies
d=dict()
for i in temp:
    d[i]=test_str.count(i)

#20- Possible Substring count from String
def count_substrings(test_str, arg_str):
    """
    Counts how many times arg_str can be constructed from test_str, 
    without repeating characters from test_str.

    Args:
        test_str: The string to search within.
        arg_str: The substring to construct.

    Returns:
        The number of times arg_str can be constructed.
    """
    count = 0
    test_str_list = list(test_str)  # Convert to list for easy removal

    while True:
        temp_test_str = test_str_list[:] #create a copy of the list to work with
        found = True
        for char in arg_str:
            if char in temp_test_str:
                temp_test_str.remove(char)
            else:
                found = False
                break

        if found:
            count += 1
            #Update the main list with any remaining characters.
            test_str_list = temp_test_str[:]
        else:
            break

    return count

#21-Sandwiched_Vowels
'''
For a given string s comprising only lowercase English alphabets, eliminate the vowels from the string that occur between two consonants(sandwiched between two immediately adjacent consonants). Return the new string.

Examples:

Input : s = "bab"
Output : bb
Explanation: 'a' is a vowel occuring between two consonants i.e. b. Hence the updated string eliminates a.
'''
def Sandwiched_Vowel(s):
    #Complete the function
    vowels="aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"

    res=""
    for i in range(len(s)):
        if s[i] in vowels :
            if i > 0 and i < len(s) - 1 and s[i - 1] in consonants and s[i + 1] in consonants:
                continue
            else:
                res+=s[i]
        else:
            res+=s[i]
    return res

#22-Substrings with same first and last characters
'''
Given string s, the task is to find the count of all substrings which have the same character at the beginning and end.

Example 1:

Input: s = "abcab"
Output: 7
Explanation: a, abca, b, bcab, 
c, a and b
Example 2:

Input: s = "aba"
Output: 4
Explanation: a, b, a and aba
'''
from collections import Counter
class Solution:
	def countSubstringWithEqualEnds(self, s):
        return sum(f * (f + 1) // 2 for f in Counter(s).values())

#23-Floating point number even or odd
'''
Given a floating point number in string format s, check whether it is even or odd.
Example 1:
Input: 
n = 4
s = 97.8
Output: EVEN
Explanation: 8 is even number.
'''
    def isEven(self, s, n):
        s = s.rstrip("0")
        s = s.rstrip(".")
        n = int(s[-1])
        return  n & 1 ==0
#24-No of Carry Operations
'''
Input:
A = 1234
B = 5678
Output: 
2
Explanation:
1234
+
5678
--------
6912
--------
4+8 = 2 and carry 1
carry+3+7 = carry 1
carry+2+6 = 9, carry 0
carry+1+5 = 6

So, there are 2 Carry Operations.
'''
def count_carry(A,B):
	A_str,B_str=str(A),str(B)
	carry,carr_count=0,0
	for i in range(len(A_str)-1,-1,-1):
		digit_A=int(A_str[i])
		digit_B=int(B_str[i])
		total=Digit_A+digit_B+carry
		if total>9:
			carry=1
			carry_count+=1
		else:
			carry=0
	return carry_count
#25-Minimize string value
'''
Given a string of lowercase alphabets and a number k, the task is to find the minimum value of the string after removal of ‘k’ characters. 
The value of a string is defined as the sum of squares of the count of each distinct character.
For example consider the string “geeks”, here frequencies of characters are g -> 1, e -> 2, k -> 1, s -> 1 and value of the string is 12 + 22 + 12 + 12 = 7
Example 1:
Input: S = "abccc", K = 1
Output: 6
Explanation: Remove one 'c', then frequency
will be a -> 1, b -> 1, c -> 2.
12 + 12 + 22 = 6
'''
from collections import Counter
	def minValue(self, S, K):
        freq = Counter(S)
        freq_list = sorted(freq.values(), reverse=True)

        for _ in range(K):
            if freq_list[0] > 0:
                freq_list[0] -= 1
            freq_list.sort(reverse=True)

        res = sum(f ** 2 for f in freq_list)
        return res
#26-Max-Min conversion
'''
Given a number N. You can perform an operation on it multiple times, in which you can change digit 5 to 6 and vice versa.
You have to return the sum of the maximum number and the minimum number which can be obtained by performing such operations.

Example 1:

Input: N = 35
Output: 71
Explanation: The maximum number which can be
formed is 36 and the minimum number which can
be formed is 35 itself. 
'''
def performOperation(N):
    #code here
    N_str = str(N)
    max_str = ""
    min_str = ""

    for digit in N_str:
        if digit == '5':
            max_str += '6'
            min_str += '5'
        elif digit == '6':
            max_str += '6'
            min_str += '5'
        else:
            max_str += digit
            min_str += digit

    max_num = int(max_str)
    min_num = int(min_str)

    return max_num + min_num
#27-replace the bit
'''
Given two numbers N and K, change the Kth (1-based indexing) bit from the left of the binary representation of the number N to '0' if it is  '1', else return the number N itself.

Example 1:

Input:
N = 13, K = 2
Output: 9
Explanation: Binary representation of 13 is
1101. The 2nd bit from left is 1, we make
it 0 and result is 1001 = 9 (decimal).
'''



    def replaceBit(self, N, K):
        # code here
        
        binary = bin(N)[2:]
        
        binary_list = list(binary)
        
        index_from_left = K - 1 
        
        if K > len(binary_list):
            return N
        else:
            if binary_list[index_from_left] == '1':
                binary_list[index_from_left] = '0'
                
            return int(''.join(binary_list),2)
#29-Repeating Character - First Appearance Leftmost
'''
You are given a string S (both uppercase and lowercase characters). You need to print the index of repeated character whose first appearance is leftmost.

Example 1:

Input:
S = geeksforgeeks
Output: 0
Explanation: We see that both e and g
repeat as we move from left to right.
But the leftmost is g so we print g.
'''
    def repeatingCharacter(self,s):
        #code here
        store = {}
        repeatingChars = []
        for idx,ch in enumerate(s):
            if ch in store:
                repeatingChars.append((ch, store[ch]))
            else:
                store[ch] = idx
        
        return sorted(repeatingChars, key=lambda x: x[1])[0][1] if repeatingChars else -1

#30-return most common char
    def getMaxOccurringChar(self, s):
        #code here
        if not s:
            return None
    
        char_counts = Counter(s)
        max_count = 0
        result = ''
    
        for char, count in char_counts.items():
            if count > max_count:
                max_count = count
                result = char
            elif count == max_count:
                result = min(result, char)
    
        return result

#31-Extraction of secret message
'''
You are given an encoded string S of length N. The encoded string is mixed with some number of substring "LIE" and some secret message. You have to extract secret message from it by removing all the "LIE" substrings.
For example - "I AM COOL" is given as "LIEILIEAMLIELIECOOL".

Example 1:

Input: S = "LIEILIEAMLIELIECOOL"
Output: "I AM COOL"
'''
    def ExtractMessage(self, S):
        # code here
        result=s.replace("LIE"," ")
        return " ".join(result.split())
#32-Encrypt the string - 1

'''
Bingu was testing all the strings he had at his place and found that most of them were prone to a vicious attack by Banju, his arch-enemy. Bingu decided to encrypt all the strings he had, by the following method. Every substring of identical letters is replaced by a single instance of that letter followed by the number of occurrences of that letter. Then, the string thus obtained is further encrypted by reversing it.

Example 1:

Input:
s = "aabc"
Output: 1c1b2a
Explanation: aabc
Step1: a2b1c1
Step2: 1c1b2a
'''
    def encryptString(self, s):
        a=""
        count=0
        n=len(s)
        for i in range(n-1):
            if(s[i]==s[i+1]):
                count+=1
            else:
                a+=s[i]
                a+=str((count+1))
                count=0
        a+=s[n-1]
        a+=str((count+1))
        # if(s[n-1]==s[n-2]):
        #     a+=(count+1)
        # else:
        return(a[::-1])

31-longest substring containing '1'

def maxlength(s):
    
    #add code here
    max_length = 0
    current_length = 0

    for char in s:
        if char == '1':
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0  # Reset counter when '0' is encountered

    return max_length

#32-Maximum Integer Value
'''
Given a string S of digits(0-9).Find the maximum value that can be obtained from the string by putting either '*' or '+' operators in between the digits while traversing from left to right of the string and picking up a single digit at a time.
'''
    def MaximumIntegerValue(self,S):
        #code here
        if not S:
            return 0
    
        result = int(S[0])
        for i in range(1, len(S)):
            digit = int(S[i])
            result = max(result + digit, result * digit)
    
        return result
#33-Binary String
'''
Given a binary string S. The task is to count the number of substrings that start and end with 1. For example, if the input string is “00100101”, then there are three substrings “1001”, “100101” and “101”.

Example 1:

Input:
N = 4
S = 1111
Output: 6
Explanation: There are 6 substrings from
the given string. They are 11, 11, 11,
111, 111, 1111.
'''
    def binarySubstring(self,n,S):
        #code here
        count = 0
        ones_count = 0
        for char in S:
            if char == '1':
                ones_count += 1
        return ones_count * (ones_count - 1) // 2

#33-Perfect Square String
'''
You are given a string S. Your task is to determine if the sum of ASCII values of all character results in a perfect square or not. If it is a perfect square then the answer is 1, otherwise 0.

Example 1:

Input: S = "d"
Output: 1
Explanation: Sum of ASCII of S is 100 
which is a square number.

'''
    def isSquare(self, S):
        # Calculate the sum of ASCII values of all characters
        ascii_sum = sum(ord(char) for char in S)
        
        # Check if the sum is a perfect square
        root = int(math.sqrt(ascii_sum))
        if root * root == ascii_sum:
            return 1
        else:
            return 0
#34-delete odd charcter
	s=[::2]
#35-Merge two strings
'''
Given two strings S1 and S2 as input, the task is to merge them alternatively i.e. the first character of S1 then the first character of S2 and so on till the strings end.
NOTE: Add the whole string if other string is empty.
Example 1:
Input:
S1 = "Hello" S2 = "Bye"
Output: HBeylelo
Explanation: The characters of both the 
given strings are arranged alternatlively.
'''
    def merge(self, S1, S2):
        # code here
        merged = ""
        i, j = 0, 0
        while i < len(S1) and j < len(S2):
            merged += S1[i] + S2[j]
            i += 1
            j += 1
    
        # Add remaining characters from S1
        if i < len(S1):
            merged += S1[i:]
    
        # Add remaining characters from S2
        if j < len(S2):
            merged += S2[j:]
    
        return merged
#36-Count number of equal pairs in a string
'''
Given a string, find the number of pairs of characters that are same. Pairs (s[i], s[j]), (s[j], s[i]), (s[i], s[i]), (s[j], s[j]) should be considered different.

Example 1:

Input:
S = "air"
Output: 3
Explanation: 3 pairs that are equal:
(S[0], S[0]), (S[1], S[1]) and (S[2], S[2])
â€‹Example 2:

Input: 
S = "aa"
Output: 4
Explanation: 4 pairs that are equal:
(S[0], S[0]), (S[0], S[1]), (S[1], S[0])
and (S[1], S[1])
'''
class Solution:
    def equalPairs (self,S):
        # your code here
        freq = Counter(s)
        total_pairs = 0
        for char, count in freq.items():
            total_pairs += count * count
        return total_pairs
#37-Check for subsequence
'''
Given two strings A and B, find if A is a subsequence of B.

Example 1:

Input:
A = AXY 
B = YADXCP
Output: 0 
Explanation: A is not a subsequence of B
as 'Y' appears before 'A'.
'''
    def isSubSequence(self, A, B):
        #code here
        i = 0  # Pointer for string A
        j = 0  # Pointer for string B
    
        while i < len(A) and j < len(B):
            if A[i] == B[j]:
                i += 1
            j += 1
    
        return 1 if i == len(A) else 0

#38-Odd to Even
'''
Given an odd number in the form of string, the task is to make largest even number possible from the given number provided one is allowed to do exactly only one swap operation, if no such number is possible then return the input string itself.

Example 1:

Input:
s = 4543
Output: 4534
Explanation: Swap 4(3rd pos) and 3.
 

Example 2:

Input:
s = 1539
Output: 1539
Explanation: No even no. present.
'''
    def makeEven(self, s):
        # code here
        last=int(s[-1])
        index=-1
        for i in range(len(s)):
            if int(s[i])%2==0:
                index=i
                if int(s[index])<last:
                    break
        if index==-1:
            return s
        s=list(s)   
        s[index],s[-1]=s[-1],s[index]
        return ''.join(s)        # code here
#39-Extract the integers
'''
Given a string s, extract all the integers from s.

Example 1:

Input:
s = "1: Prakhar Agrawal, 2: Manish Kumar Rai, 
     3: Rishabh Gupta56"
Output: 1 2 3 56
Explanation: 
1, 2, 3, 56 are the integers present in s.
'''
    def extractIntegerWords(self, s):
        num = []
        n = ''
        for i in s:
            # Checking i is digit or not
            # if digit the add i into n string
            if i.isdigit():
                n += i
            # if i not digit
            # then checking n is empty or not
            # if not then append n into num
            elif n:
                num.append(n)
                n = ''
        # Cheking n is empty or not after the loop
        # if not empty then append n into num
        if n:
            num.append(n)
        return num

#40-Binary Addition of 1
'''
You are given a binary string s of length n. You have to perform binary addition of the string with '1'.

 

Example 1:

Input: 
n = 4
s = 1010
Output: 1011
Explaination: 
The decimal equivalent of given s is 10, 
Adding 1 gives 11, its binary representation
is 1011.
 

Example 2:

Input: 
n = 3
s = 111
Output: 1000
Explaination: The given number is 7. 
Now 7+1 = 8, whose binary representation 
is 1000.
'''
    def binaryAdd(self, n, s):
            # code here
        decimal=int(s,2)
        decimal+=1
        res=bin(decimal)[2:]
        pad=res.zfill(n)
        return pad

#41-Remove repeated digits in a given number
'''
Given an integer N represented in the form of a string, remove consecutive repeated digits from it.

Example 1:

Input:
N = 1224
Output: 124
Explanation: Two consecutive occurrences of 
2 have been reduced to one.
â€‹Example 2:

Input: 
N = 1242
Output: 1242
Explanation: No digit is repeating 
consecutively in N.
'''
    def modify(self, N):
        #code here
        N=str(N)
        l=list(N)
        ans=''
        for i in range(len(l)-1):
            if l[i]==l[i+1]:
                continue
            else:
                ans+=l[i]
        ans+=l[-1]
        return ans
#42-Good String
'''
Given a string s of length N, you have to tell whether it is good or not. A good string is one where the distance between every two adjacent character is exactly 1. Here distance is defined by minimum distance between two character when alphabets from 'a' to 'z' are put in cyclic manner. For example distance between 'a' to 'c' is 2 and distance between 'a' to 'y' is also 2. The task is to return "YES" or "NO" (without quotes) depending on whether the given string is Good or not.

Note: Unit length string will be always good.

Example 1:

Input: s = "aaa"
Output: NO
Explanation: distance between 'a' and 'a' is not 1.
Example 2:

Input: s = "cbc"
Output: YES
Explanation: distance between 'b' and 'c' is 1.
'''
    def isGoodString(self, s):
        n = len(s)
        
        if n == 1:
            return "YES"
        
        for i in range(n-1):
            diff = abs(ord(s[i]) - ord(s[i + 1]))
            cyclic_diff = min(diff, 26 - diff)
            
            if cyclic_diff !=1:
                return "NO"
            
        return "YES"
#43-Check if divisible by 11

'''
Given a number S. Check whether it is divisble by 11 or not.

Example 1:

Input:
S = 76945
Output: 1
Explanation: The number is divisible by 11
as 76945 % 11 = 0.

â€‹Example 2:

Input: 
S = 12
Output: 0
Explanation: The number is not divisible
by 11 as 12 % 11 = 1.
'''
	def divisibleBy11(self, S):
		# Your Code Here
    	n = len(S)
        alternate_sum = 0
        for i in range(n):
            digit = int(S[i])
            if i % 2 == 0:
                alternate_sum += digit
            else:
                alternate_sum -= digit
    
        if alternate_sum % 11 == 0:
            return 1
        else:
            return 0
#44-Crazy String
'''
You have given a non-empty string. This string can consist of lowercase and uppercase english alphabets. Convert the string into an alternating sequence of lowercase and uppercase characters without changing the character at the 0th index.

Example 1:

Input:
S = "geeksforgeeks"
Output: gEeKsFoRgEeKs
Explanation: The first character is kept
unchanged whereas all the characters are
arranged in alternating order of lowercase
and uppercase.
'''
    def getCrazy(self, S):
        # code here
        result=[]
        result.append(S[0])
        if S[0].islower():
            for i in range(1,len(S)):
                if i%2==0:
                    result.append(S[i].lower())
                else:
                    result.append(S[i].upper())
        else:
            for i in range(1,len(S)):
                if i%2==1:
                    result.append(S[i].lower())
                else:
                    result.append(S[i].upper())
       
        return "".join(result) 
#45-Upper Case Conversion
'''
Given a string s, convert the first letter of each word in the string to uppercase. 

Examples:

Input: s = "gEEKs"
Output: "Geeks"
Input: s = "i love programming"
Output: "I Love Programming"
'''
    def convert(self, s):
        return ' '.join(word.capitalize() for word in s.split())

#46-Pangram Checking
'''
You are given a string s. You need to find if the string is a panagram or not.

A panagram contains all the letters of english alphabet at least once.

Examples:

Input: s = "Thequickbrownfoxjumpsoverthelazydog"
Output: 1
Input: s = "HeavyDuty"
Output: 0
'''
    def isPanagram(self,s):
        s = s.lower()
        alphabet = set("abcdefghijklmnopqrstuvwxyz")
        char_set = set()
    
        for char in s:
            if 'a' <= char <= 'z':
                char_set.add(char)
    
        if char_set == alphabet:
            return 1
        else:
            return 0

#47-Print first letter of every word in the string
'''
Given a string S, the task is to create a string with the first letter of every word in the string.
 

Example 1:

Input: 
S = "geeks for geeks"
Output: gfg

Example 2:

Input: 
S = "bad is good"
Output: big
'''
	def firstAlphabet(self, S):
		words=S.split()
        result=""
        for word in words:
            result += word[0]
        return result
#47-Check Binary String
'''
Given a binary string S of 0 and 1, check if the given string is valid or not. The given string is valid when there is no zero is present in between 1s.

Example 1:

Input:
S = "100"
Output: VALID
Explanation: 100. The string have just a
single 1, hence it is valid.

'''
    def checkBinary (self, s):
        # Your code here
        s = str(int(s))
        for i in range(len(s)-1):
            if s[i] == '0' and s[i+1] == '1':
                return False
        return True

#50-Anagram of String
'''
Given two strings s1 and s2 in lowercase, the task is to make them anagrams. The only allowed operation is to remove a character from any string. Find the minimum number of characters to be deleted to make both the strings anagram. Two strings are called anagrams of each other if one of them can be converted into another by rearranging its letters.

Examples:

Input: s1 = "bcadeh", s2 = "hea"
Output: 3
Explanation: We need to remove b, c and d from s1.
Input: s1 = "cddgk", s2 = "gcd"
Output: 2
Explanation: We need to remove d and k from s1.
'''
from collections import Counter
def remAnagram(s1,s2):
    
    freq1 = Counter(s1)
    freq2 = Counter(s2)

    deletions = 0
    all_chars = set(freq1.keys()) | set(freq2.keys()) #added this line to handle characters unique to s2

    for char in all_chars:
        deletions += abs(freq1.get(char,0) - freq2.get(char, 0)) #get with default value of 0

    return deletions    

#51-Confused pappu
'''
Pappu is confused between 6 & 9. He works in the billing department of abc company and his work is to return the remaining amount to the customers. If the actual remaining amount is given we need to find the maximum possible extra amount given by the pappu to the customers.

Example 1:

Input: amount = 56
Output: 3
Explanation: maximum possible extra 
             amount = 59 - 56 = 3.
Example 2:

Input: amount = 66
Output: 33
Explanation: maximum possible extra 
             amount = 99 - 66 = 33.
'''
    def findDiff(self, amount):
        amount_str = str(amount)
        max_amount_str = amount_str.replace('6', '9')
        max_amount = int(max_amount_str)
        return max_amount - amount

#52-Remainder with 7
'''
Given a number as string(n) , find the remainder of the number when it is divided by 7

Example 1:

Input:
5
Output:
5
'''
    def remainderWith7(self, str):
        #Code here
      remainder = 0
      for digit in str:
        remainder = (remainder * 10 + int(digit)) % 7
      return remainder
#53-Magical String[Duplicate Problem]
'''
You are given a string S, convert it into a magical string.
A string can be made into a magical string if the alphabets are swapped in the given manner: a->z or z->a, b->y or y->b, and so on.  
 

Note: All the alphabets in the string are in lowercase.

 

Example 1:

Input:
S = varun
Output:
ezifm
Explanation:
Magical string of "varun" 
will be "ezifm" 
since v->e , a->z , 
r->i , u->f and n->m.
'''
    def magicalString (ob,s):
        # code here 
        magic_str = ""
        for char in s:
            if 'a' <= char <= 'z':
                magic_str += chr(ord('z') - (ord(char) - ord('a')))
            else:
                magic_str += char  # If not a lowercase letter, keep the same
    
        return magic_str






















