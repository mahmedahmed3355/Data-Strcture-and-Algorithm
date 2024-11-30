#two sum where input array is sorted
def twosum(numbers,target):
    l,r=0,len(numbers)-1
    while l<r:
        if numbers[l]+numbers[r]==target:
            return[l+1,r+1]
        elif numbers[l]+numbers[r]<target:
            l+=1
        else:
            r-=1
    return []

#2- 3 sum
def threesum(nums):
    h,n,s={},len(nums),set()
    for i,um in enumerate(nums):
        h[nums]=i
    for i in range(nums):
        for j in range(i+1,n):
            desired=-nums[i]-nums[j]
            if desired in h and h[desired]!=i and h[desired]!=j:
                s.add(tuple(sorted(nums[i],nums[j],desired)))
    return s

#3-4 sum
def four_sum(nums,target):
    if not nums or len(nums)<4: return []
    nums.sort()
    res,n=set(),len(nums)
    for i in range(n):
        for j in range(i+1,n):
            l,r=j+1,n-1
            if target==nums[i]+nums[j]+nums[r]+nums[l]:
                res.add(nums[i],nums[j],nums[r],nums[l])
                l+=1
                r-=1
            elif target>nums[i]+nums[j]+nums[r]+nums[l]:
                l+=1
            else:
                r-=1
    return res

#4-number of subsequence that satisfy the given sum condition
def numofsubsequence(nums,target):
    ans=0
    nums.sort()
    i,j=0,len(nums)-1
    while(i<=j):
        if nums[i]+nums[j]<=target:
            ans+=pow(2,(j-i),1000000007)
            i+=1
        else:
            j-=1
    return ans%1000000007

#5-sum of square numbers
from math import isqrt,sqrt
def judgesquaresum(c):
    for a in range(isqrt(c)+1):
        b2=c-a*a
        if sqrt(b2)%1==0:
            return True
    return False

#6- boat to save people
def numrescueboats(people,limt):
    people.sort()
    left,right=0,len(people)-1
    boats=0
    while left<=right:
        if people[left]+people[right]<=limt:
            left+=1
        right-=1
        boats+=1
    return boats

#7-minimize maximum pair sum in array
def minpairsum(nums):
    nums.sort()
    mid=len(nums)//2
    return max(a+b for a,b in zip(nums[:mid],nums[:mid-1:-1]))      

#8- 3sum with multiplicty
from collections import Counter
def threesummulti(arr,target):
    n,toreturn=len(arr),0
    nums=Counter(arr[:1])
    for j in range(1,n-1):
        for k in range(j+1,n):
            toreturn=(toreturn+nums[target-arr[k]-arr[j]])%1000000007
        nums[arr[j]]+=1
    return toreturn

#9-Next permutation
def nextpermutation(nums):
    i=len(nums)-2
    while i>=0 and nums[i]>=nums[i+1]:
        i-=1
    if i>=0:
        j=len(nums)-1
        while nums[j]<=nums[i]:
            j-=1
        nums[i],nums[j]=nums[j],nums[i]
    nums[i+1:]=reversed(nums[i+1:])

#10-next greater element
def nextgreaterelement(n):
    cs=list(str(n))
    n,i,j=len(cs),n-2,n-1
    while i>=0 and cs[i]>=cs[i+1]:
        i-=1
    if i<0:
        return -1
    while cs[i]>=cs[j]:
        j-=1
    cs[i],cs[j]=cs[j],cs[i]
    cs[i+1:]=cs[i+1:][::-1]
    ans=int("".join(cs))
    return -1 if ans>2**31-1 else ans

#11- longest substring without repeating charcter
def longestsubstring(s):
    seen,l,max_length =set(),0,0
    for r,char in enumerate(s):
        while char in seen:
            seen.remove(s[l])
            l+=1
        seen.add(char)
        max_length=max(max_length,r-l+1)
    return max_length

#12-longest substring with at most two distinct charcter
def longestsubstringwithdistinictcharcter(s):
    seen,ans,max_len=set(),"",-1
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            seen=set(s[i:j])
            if len(seen)==2:
                max_len=max(max_len,j-i)
    return max_len

#13-longest consequative subcharcter
def solution(s):
    maxx,count=1,1
    for i in range(len(s)):
        if s[i]==s[i+1]:
            count+=1
            maxx=max(maxx,count)
        else:
            count=1
    return maxx

#14-longest increasing subsequence
def lis(nums):
    if not nums : return 0
    res,curr=0,0
    for i in range(len(nums)):
        if nums[i]>nums[i-1]:
            curr+=1
            res=max(res,curr)
        else:
            curr=1
    return res

#15-reverse string with special charcter
def reversewitjspecialcharcter(s):
    s,r,l=list(s),0,len(s)-1
    while l<r:
        if not s[l].isalpha():
            l+=1
        if not s[r].isalpha():
            r-=1
        else:
            s[l],s[r]=s[r],s[l]
            l+=1
            r-=1
    return "".join(s)
    
#16-reverse words in string
def reversewordinstring(s):
    return "".join(reversed(s.split()))

#17-reverse word in string 2
def refersewordsinstring2(s):
    def reverse(i,j):
        while i<j:
            s[i],s[j]=s[j],s[i]
            i,j=i+1,j-1
    i,n=0,len(s)
    for j,c in enumerate(s):
        if c=="":
            reverse(i,j-1)
            i=i+1
        elif j==n-1:
            reverse(i,j)
    reverse(0,n-1)

#18-reverse words in string3
def reversewords(s):
    return "".join(word[::-1] for word in s.split(""))

#19-reverse string
def reversestring(s):
    l,r=0,len(s)-1
    while l<r:
        s[l],s[r]=s[r],s[l]
        l,r=l+1,r-1

#20-reverse string 2
def reversestr(s,k):
    lst=list(s)
    for i in range(0,len(lst),2*k):
        lst[i:i+k]=reversed(lst[i:i+k])
    return "".join(lst)

#21-reverse vowels in string
def revesevowels(s):
    vowels="AEIOUaeiou"
    vews=[c for c in s if c in vowels]
    revstr=[]
    for char in s:
        if char in vowels:
            revstr.append(vews.pop())
        else:
            revstr.append(char)
    return "".join(revstr)

#22-reverse only letters
def reverse_only_letters(s):
    cs=list(s)
    i,j=0,len(cs)-1
    while i<j:
        while i<j and not cs[i].isalpha():
            i+=1
        while i<j and not cs[i].isalpha():
            j-=1
        if i<j:
            cs[i],cs[j]=cs[j],cs[i]
            i,j=i+1,j-1
    return "".join(cs)

#23-reverse string between each pair of parenthes
def reversepathrness(s):
    stk=[]
    for c in s:
        if c==")":
            t=[]
            while stk[-1]!="(":
                t.append(stk.pop())
            stk.pop()
            stk.append(t)
        else:
            stk.append(c)
    return "".join(stk)

#24-reverse prefix in string
def reverseprefixword(word,ch):
    i=word.find(ch)
    if i !=-1:
        return word[:i+1][::-1]+word[i+1:]
    return word

#25-revere words with dot
def reverse_words(str):
    input_word=str.split(".")
    reversed_word=input_word[::-1]
    return ".".join(reversed_word)

#26-reverse string
def reverseinteger(x):
    sign=-1 if x<0 else 1
    x*=sign
    reversed_x=int(str(x)[::-1])
    reversed_x*=sign
    if reversed_x<-2**31 or reversed_x>2**31-1:
        return 0
    return reversed_x

#27-vaild palindrome
def ispalindrome(s):
    s=s.lower()
    s="".join(filter(str.isalnum(s)))
    return 1 if s==s[::-1] else 0

#28-vaild palindrome 2
def vaildpalindrome(s):
    if s==s[::-1] : return True
    l,r=0,len(s)-1
    while l<=r:
        if s[l]!=s[r]:
            temp2=s[:r]+s[r+1:]
            temp=s[:l]+s[l+1:]
            return temp2==temp[::-1] or temp[::-1]==temp2[::-1]
        l+=1
        r-=1

#29-vaild palindrome3
# mak string palindrome by removinr one char return T/F
def makepalindrome(s):
    i,j=0,len(s)-1
    cnt=0
    while i<j:
        cnt+=s[i]!=s[j]
        i,j=i+1,j-1
    return cnt<=2
#30-longest palindrome substring
def longest_palindrome(s):
    longest,i,j="",0,0
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            if s[i:j]==s[i:j][::-1]:
                if len(s[i:j])>len(longest):
                    longest=s[i:j]
    return longest

#31-count palindrome substring
def countpalindrome(s):
    counter,i,j=0,0,0
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            if s[i:j]==s[i:j][::-1]:
                counter+=1
    return counter

#32-convert to palindrome
def slove(a):
    l,r=0,len(a)-1
    while l<r:
        if a[l]==a[r]:
            l+=1
            r-=1
        else:
            if a[l+1:r+1]==a[l+1:r+1][::-1]:
                return 1
            if a[l:r]==a[l:r][::-1]:
                return 1
            return 0
    return 1

#33-shortst subarray to be removed to make array sorted
def findlengthofshortestsubarray(arr):
    n,left=len(arr),0
    while left+1<n and arr[left]<=arr[left+1]:
        left+=1
    if left==n-1:
        return 0
    right=n-1
    while right>0 and arr[right-1]<=arr[right]:
        right-=1
    result=min(n-left-1,right)

#34-longest repeating substring
def longestrepeatingsubstring(s):
    n=len(s)
    dp=[[0]*n for _ in range(n)]
    ans=0
    for i in range(n):
        for j in range(i+1,n):
            if s[i]==s[j]:
                dp[i][j]=dp[i-1][j-1]+1 if i else 1
                ans=max(ans,dp[i][j])
    return ans
#35-replace all ? to avoid consequative repeating charcter
def modifystring(s):
    char_list=list(s)
    n=len(s)
    for i in range(n):
        if char_list[i]=="?":
            for c in "abc":
                if (i>0 and char_list[i-1]==c)or (i+1<n and char_list[i+1]==c):
                    continue
            char_list[i]=c
            break
    return "".join(char_list)

#36-the length of longest substring without repeating charcter
def lengthoflongestsubstring(s):
    if len(s)==len(set(s)): return len(s)
    substring,maxlen="",0
    for i in s:
        if i not in substring:
            substring=substring+i
            maxlen=max(maxlen,len(substring))
        else:
            substring=substring.split(i)[1]+i
    return maxlen

#37-maximum repeating substring
def maxrepeating(sequence,word):
    if len(sequence)<len(word): return 0
    ans=0
    k=1
    while word*k in sequence:
        ans+=1
        k+=1
    return ans

#38-count sunstring without repeating charcter
def count_substring(s):
    count=0
    for i in range(len(s)):
        for j in range(i+1,len(s)):
            substring=s[i:j+1]
            if len(substring)==len(set(substring)):
                count+=1
    return count

#39-longest uncommon subsequence
def longestuncommon(a,b):
    if a==b: return -1
    else:
        return max(len(a),len(b))

#40-split string to max number unique string
def maxunique(s):
    global d
    maxm=0
    for i in range(1,len(s)+1):
        tmp=s[0:i]
        if tmp not in d:
            d[tmp]=1
            maxm=max(maxm,maxunique(s[i:])+1)
            del d[tmp]
    return maxm

#41- magical string
def magicalstring(n):
    s=["1","2","2"]
    for i in range(2,n):
        add_two=s[-1]=='-1'
        s.extend(list(int(s[i])*("2" if add_two else "1")))
        if len(s)>=n: break
    return s[:n].count('1')

#42- merge sorted array
def merge(nums1,m,nums2,n):
    t=m+n
    keyy=[]
    for i in (nums1[:m]):
        keyy.append(i)
    for i in (nums2[:n]):
        keyy.append(i)
    keyy.sort()
    nums1[:t]=keyy[:t]

#43-findthe distance value
def findthedistance(arr1,arr2,d):
    s=0
    for i in arr1:
        t=0
        for j in arr2:
            if abs(i-j)<=d:
                t=1
                break
        if t==0:
            s=s+1
    return s

#44- mergee intervals
def merge(intervals):
    if len(intervals)<=1: return intervals
    intervals.sort()
    res=[intervals[0]]
    for start,end in intervals[1:]:
        if start<=res[-1][-1]:
            res[-1][1]=max(end,res[-1][1])
        else:
            res.append([start,end])
    return res

#45-insert intervals
def insert(intervals,newintervals):
    s,e=newintervals.start,newintervals.end
    left,right=[],[]
    for i in intervals:
        if i.end<s:
            left+=i
        elif i.start>e:
            right+=i
        else:
            s=min(s,i.start)
            e=max(e,i.end)
    return left+[intervals(s,e)]+right

#46-none overlapping intervals
def erase_overlaping_intervals(intervals):
    intervals.sort()
    ans=0
    prevend=intervals[0][1]
    for start,end in intervals[1:]:
        if start>=prevend:
            prevend=end
        else:
            ans+=1
            prevend=min(end,prevend)
    return ans

#47-maximum points you can obtain from cards
def maxcore(cardpoints,k):
    dparray=[0 for i in range(k+1)]
    dparray[0]=sum(cardpoints[:k])
    for i in range(1,k+1):
        dparray[i]=dparray[i-1]-cardpoints[k-i]+cardpoints[-i]
    return max(dparray)

#48-Maximum Erasure Value
def maximumuniquesubarray(nums)
    s,suml,start,m=set(),0,0,0
    for i in range(len(nums)):
        while nums[i] in s:
            s.remove(nums[start])
            suml-=nums[start]
            start+=1
        s.add(nums[i])
        suml+=nums[i]
        m=max(m,suml)
    return m

#49 - number of substring containg all 3 chrcter
def numberofsubstring(s):
    l={'a':-1,'b':-1,'c':-1}
    ans=0
    for i,ch in enumerate(s):
        l[ch]=i
        ans+=min(l.values())+1
    return ans
#50-square of sorted array
def sortedsquares(nums):
    nums=[i*i for i in nums]
    nums.sort()
    return nums

#51-max consequative ones
def longestones(nums,K):
    if not nums : return 0
    max_0,c=0,0
    for i in nums:
        if i ==1:
            c+=1
        else:
            max_o=max(c,max_o)
            c=0
    return max(c,max_o)

#52-max consequative one 
def findmaxconsequativeones(nums):
    max_count=0
    count=0
    for i in nums:
        if i==1:
            count+=1
        else:
            max_count=max(max_count,count)
            count=0
    return max(max_count,count)








