#1-Recursion  Patterns
#2. Tail Recursion
'''
The recursive call is the last operation in the function, which can be optimized by some compilers.
Question:
Convert Sorted List to Binary Search Tree
Convert a sorted linked list to a height-balanced binary search tree.
'''
#3-Divide and Conquer
'''
The problem is divided into smaller subproblems, solved independently, and the results are combined.
Questions:
Merge Sort (Sort an Array)
Find Kth Largest Element in an Array
'''
#4-Backtracking
'''
Incrementally builds candidates for solutions and abandons them when they are determined not to be valid.
Questions:
N-Queens
Subset Sum
'''
#5-Dynamic Programming (Memoization)
'''
Stores results of expensive function calls and reuses them when the same inputs occur.
Questions:
Climbing Stairs
Longest Increasing Subsequence
'''
#6-Tree Traversal
'''
Recursively visits nodes in tree data structures.
Questions:
Binary Tree Inorder Traversal
Binary Tree Preorder Traversal
'''
#7. Combinations and Permutations
'''
Generating all combinations or permutations of a set of elements.
Questions:
Combinations
Permutations
'''

#8. Depth-First Search (DFS)
'''
A traversal method for searching tree or graph data structures.
Questions:
Number of Islands
Course Schedule
'''

#9. Generating Subsets
'''
Creating all possible subsets of a set.
Question:
Subsets
Given an integer array, return all possible subsets.
'''

#10. Matrix Recursion
'''
Problems involving traversing or manipulating matrices.
Questions:
Word Search
Number of Unique Paths
'''

#11. String Manipulation
'''
Solving problems involving strings using recursion.
Questions:
Palindrome Partitioning
Generate Parentheses
'''

#12. Knapsack Problem
'''
Solving variations of the knapsack problem using recursion and dynamic programming techniques.
Question:
0/1 Knapsack Problem
'''

#13. Graph Algorithms
'''
Solving problems involving graphs.
Questions:
Clone Graph
Minimum Height Trees
'''

#14. Recursive Descent Parsing
'''
A method of parsing expressions in compilers.
Question:
Basic Calculator
'''

#15. Breadth-First Search (BFS) with Recursion
'''
Although typically implemented using queues, BFS can also be implemented recursively.
Question:
Binary Tree Level Order Traversal
'''

#1-series sum
def series_sum(n):
    if n==0: return 1
    else:
        return n+n*series_sum(n-1)

#2- Subset 
'''
Given an integer array nums of unique elements, return all possible subsets (the power set).
'''
def subsets(nums):
    result=[[]]
    for num in nums:
        new_subset=[]
        for subset in result:
            new_subset.append(subset+[num])
        result.extend(new_subset)
    return result
#3- permutations
'''
Given an array nums of distinct integers, return all the possible permutations. 
You can return the answer in any order.
'''
from itertools import permutations
def permute(nums):
    return list(permutations(nums))

#4-combintations
'''
Given two integers n and k, return all possible combinations of k numbers chosen 
from the range [1, n].
'''
from itertools import combinations
def combine(n,k):
    return list(combinations(range(1,n+1),k))

#5- combination sum
def combinationsum(candidate,target):
    ans=[]
    def dfs(s,target,path):
        if target<0: return 
        if target==0:
            ans.append(path[:])
            return 
        for i in range(s,len(candidate)):
            path.append(candidate[i])
            dfs(i,target-candidate[i],path)
            path.pop()
    candidate.sort()
    dfs(0,target,[])
    return ans

#6- print the number from 1 to N
def print1toN(n):
    if n==0 : return 
    print1toN(n-1)
    print(n)
#7- print the element of the array recursively
def print_array_recursive(arr,n,index=0):
    if index==n: return 
    print(arr[index],end=" ")
    print_array_recursive(arr,n,index+1)

#8-find sum of its digits using recursion.
#ex: n=12345==>15
def dsum(n):
    if n<10: return n
    return dsum(n//10)+n%10
#9-Counts the number of digits in an integer
def count_digits(n):
    if n==0: return 1
    count=0
    n=abs(n)
    while n>0:
        n//=10
        count+=1
    return count

#10-factroial
def factorial_recursive(n):
    if n==0: 
        return 1
    else:
        return n*factorial_recursive(n-1)
#11-is palindrome_recursive
def is_palindrome_recursive(s):
    s="".join(c.lower() for c in s if c.isalnum())
    if len(s)<=1: 
        return True
    if s[0]==s[-1]:
        return is_palindrome_recursive(s[1:-1])
    else:
        return False

#12- Robe cutting
'''
Given a rod of length N meters, and the rod can be cut in only 3 sizes A, B and C. 
The task is to maximizes the number of cuts in rod. If it is impossible to make cut then print -1.
'''
def maxpeices(n,a,b,c) :
    if n == 0 :
        return 0
    if n <= -1 :
        return -1 
    res = max(maxpeices(n-a,a,b,c),
              maxpeices(n-b,a,b,c),
              maxpeices(n-c,a,b,c))
    if res == -1 :
        return -1 
    return res + 1 
#13-remove all adjacent duplicates
def remove_adjacent_duplicates(s):
    if len(s)<=1:
        return s
    if s[0]==s[1]:
        i=1
        while i<len(s) and s[0]==s[i]:
            i+=1
        return remove_adjacent_duplicates(s[i:])
    else:
        return s[0]+remove_adjacent_duplicates(s[1:])

#14-Coin change problem
def count(coins, sum):
    n = len(coins)
    
    # dp[i] will be storing the number of solutions for
    # value i. We need sum+1 rows as the dp is
    # constructed in bottom up manner using the base case
    # (sum = 0)
    dp = [0] * (sum + 1)

    # Base case (If given value is 0)
    dp[0] = 1

    # Pick all coins one by one and update the table[]
    # values after the index greater than or equal to the
    # value of the picked coin
    for i in range(n):
        for j in range(coins[i], sum + 1):
            dp[j] += dp[j - coins[i]]
            
    return dp[sum]

#15-Convert a String to an Integer using Recursion
def string_to_int_recursive(s):
    """
    Converts a string to an integer using recursion.

    Args:
      s: The input string representing an integer.

    Returns:
      The integer value of the string.
    """
    if not s:
        return 0  # Base case: empty string

    sign = 1
    if s[0] == '-':
        sign = -1
        s = s[1:]
    elif s[0] == '+':
        s = s[1:]

    if not s:
        return 0

    if not s[0].isdigit():
      return 0 #invalid input, not a digit

    if len(s) == 1:
        return sign * int(s)  # Base case: single digit

    return sign * (int(s[0]) * (10 ** (len(s) - 1)) + string_to_int_recursive(s[1:]))
#16-Print all combinations of balanced parentheses

def generate_balanced_parentheses(n):
    """
    Generates all combinations of balanced parentheses of length 2n.

    Args:
      n: The number of pairs of parentheses.

    Returns:
      A list of strings representing balanced parentheses combinations.
    """
    result = []

    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n:
            result.append(s)
            return

        if open_count < n:
            backtrack(s + "(", open_count + 1, close_count)

        if close_count < open_count:
            backtrack(s + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return result
#17-Permutations of given String

def permutations(s):
    """
    Generates all permutations of a given string in lexicographically sorted order.

    Args:
      s: The input string.

    Returns:
      A list of strings representing all permutations of s.
    """
    result = []

    def backtrack(current, remaining):
        if not remaining:  # Base case: no more characters to permute
            result.append("".join(current))
            return

        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()  # Backtrack: remove the last added character

    backtrack([], list(s))
    result.sort() #Sort the result
    return result

#18-Binary to Gray code using recursion
def binary_to_gry(n):
    if not(n): return 0
    #taking the lase digit
    a=n%10
    #taking second last digit
    b=int(n/10)%10
    if (a and not(b)) or (not(a) and not b):
        return (1+10*binary_to_gry(int(n/10)))
    return (10*binary_to_gry(int(n/10)))

#19- all possible combintation of string
import itertools
def find_subset(string):
    combintations=[]
    for i in range(len(string)+1):
        combinations+=itertools.combinations(string,i)
    subset=[]
    for c in permutations:
        subset="".join(c)
        if subset!="":
            subset.append(subset)
    return subsets

#20-Combination Sum IV
'''
Given an array of distinct integers nums and a target integer target, return the number of possible combinations 
that add up to target.
Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
'''
def combinationSum4(self, nums: List[int], target: int) -> int:
    dp = [0] * (target + 1)
    dp[0] = 1
        
    for i in range(1, target + 1):
        for num in nums:
            if i - num >= 0:
                dp[i] += dp[i - num]
                    
    return dp[target]