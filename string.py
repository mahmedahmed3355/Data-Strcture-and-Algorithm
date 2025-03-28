#1- longest substring without repeating charcter
def longest_substring_without_repeating_charcter(s):
    if not str: return 0
    r,l,seen,max_length=0,0,set(),0
    while l<len(s):
        if s[l] not in seen:
            seen.add(s[l])
            l+=1
            max_length=max(max_length,len(seen))
        else:
            seen.remove(s[r])
            r+=1
    return max_length
#2-longest substring with at most two distinct charcter
def longest_substring(s):
    seen,ans,max_len=set(),"",-1
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            seen=set(s[i:j])
            if len(seen)>=2:
                max_lem=max(max_len,i-j)
    return max_len
#3-find first non repeating charcter
def non_repeating_charcter(s):
    dict={}
    for i in s:
        if i not in dict.key():
            dict[i]+=1
        else:
            dict[i]=1
    for i in s:
        if dict[i]==1: return i
    return "0"
#4-number of substring containg three charcter
def number_os_string(s):
    chars={"a":-1,"b":-1,"c":-1}
    ans=0
    for i,c in enumerate(s):
        chars[c]=i
        ans=min(chars[c].values()+1)
    return ans
#5- consequative subcharcter
def consequative_subcharcter(s):
    if not str : return 0
    countt,maxx=1,1
    for i in range(len(s)):
        if s[i]==s[i+1]:
            countt+=1
            maxx=max(maxx,countt)
        else:
            count=1
    return maxx
#6-longest increasing subsequence
def lis(nums):
    if not nums: return 0
    res,curr=0,0
    for i in range(len(nums)):
        if nums[j]>nums[i-1]:
            curr+=1
            res=max(res,curr)
        else:
            curr=1
    return res
#7-longest common prefix
def longest_common_prefix(strs):
    import os
    if not strs : return ""
    return os.path.commonprefix(strs)
#8- the length of longest common prefix
def longest_common_prefix(arr1,arr2):
    s=set()
    for x in arr1:
        s.add(x)
        x//=10
    ans=0
    for x in arr2:
        while x:
            if x in s:
                ans=max(ans.x)
            x//10
    return len(str(ans))if ans else 0
#9-check if word occur as prefix of any word in sequence
def isprefixword(sentence,searchword):
    words=sentence.split()
    for i,word in enumerate(words):
        if word.startwith(searchword):
            return i+1
    return -1
#10- check if string is prefix of array
from itertools import accumulate
def is_prefix_string(s,words):
    return s in accumulate(words)

#11-Count Prefixes of a Given String
def countprefixes(words,s):
    return len(i for i in words if s.startwith(i))

#12-count prefix and suffix
def count_prefix_suffix(words):
    cnt=0
    for i in range(len(words)):
        for j in range(i+1,len(words)):
            if words[j].startwith(words[i])and words[j].endwith(words[i]):
                cnt+=1
    return cnt
#################################################################################################
#################################################################################################
################################################### Reverse  ###################################
#13-reverse string with specia; charcter
def reverse_string_with_special_charcter(s):
    s=list(s)
    l,r=0,len(s)
    while l<r:
        if not s[l].isalpha():
            l+=1
        elif not s[r].isalpha():
            r-=1
        else:
            s[l],s[r]=s[r],s[l]
            l+=1
            r-=1
    return "".join(s)
#14-reverse words in string
def reverse_words_in_string(s):
    return "".join(reversed(s.split()))

#15-reverse words in string2
#["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
def reverse(s):
    def reverse(i,j):
        s[i],s[j]=s[j],s[i]
        i,j=i+1,j-1
    i,n=0,len(s)
    for j,c in enumerate(s):
        if c=="":
            reverse(i,j-1)
            i=j+1
        elif j==n-1:
            reverse(i,j)
    reverse(0,n-1)
#16-reverse words in string 3
def reverse_words(s):
    return "".join(word[::-1] for word in s.split(" "))
#17-reverse string
def reverse_string(s):
    l,r=0,len(s)
    while l<r:
        s[l],s[r]=s[r],s[l]
        l+=1
        r-=1  
#18 reverse string 2
#Input: s = "abcdefg", k = 2
def reverse_str(s,k):
    lst=list(s)
    for i in range(0,len(lst),2*k):
        lst[i:i+k]=reversed(lst[i:i+k])
    return "".join(lst)       
#19 reverse vowels in string
def reverse_vowels(s):
    vowels="AEIOUaeiou"
    vows=[c for c in s if c in vowels]
    revstr=[]
    i=0
    for char in s:
        if char in vowels:
            revstr.append(vows.pop())
        else:
            revstr.append(char)
    return "".join(revstr)

#20- reverse only letters
def reverse_only_letters(s):
    cs=list(s)
    i,j=0,len(cs)-1
    while i<j:
        while i<j and not cs[i].isalpha():
            i+=1
        while i<j and not cs[j].isalpha():
            j-=1
        if i<j:
            cs[i],cs[j]=cs[j],cs[i]
            i+=1
            j-=1
    return "".join(cs)

#21-reverse string between each pair of parthness
def reverse_parthness(s):
    stack,current=[],[]
    for c in s:
        if c=="(":
            stack.append(current)
            current=[]
        elif c==")":
            current.reverse()
            current=stack.pop()+current
        else:
            current.append(c)
    return "".join(current)

#22-reverse prefix in word
def reverse_prefix(word,ch):
    i=word.find(ch)
    if i !=-1:
        return word[:i+1][::-1]+word[i+1:]
    return word
#23-reverse word with dot
def reverse_word_with_dot(str):
    input_word=str.split(".")
    reversed_word=input_word[::-1]
    return ".".join(reversed_word)

#24-reverse_integer
def reverse_integer(x):
    sign=-1 if x<0 else 1
    x*=sign
    reversed_x=int(str(x)[::-1])
    reversed_x*=sign
    if reversed_x<-2**31 or reversed_x >2**31-1:
        return 0
    return reversed_x
#####################################################################################################################################
#######################################################################################################################################
########################################################3 SubString Pattern ###########################################################
#25-single element among double
from collections import Counter
def printsingle(arr):
    freq={}
    for i in arr:
        freq[i]=freq.get(i,0)+1
    for k,v in freq.items():
        if v==1: return k
    return -1

#26-repeted DNS subsequence
def dns_subsequence(s):
    return [k for k,v in Counter(s[start:start+10]for start in range(len(s)-9)).items()if v>1]

#27-repeted string match
def repetedstringmatch(a,b):
    ans=len(b)//len(a)
    cur_str=a*ans
    for i in range(3):
        if cur_str.find(b)!=-1:
            return ans+i
        cur_str+=a
    return -1
#28-find common elements between two arrays
def find_common_elements(nums1,nums2):
    return set(nums1).intersection(set(nums2))

#29-make string repeating after every k charcter by replacing charcter in missing place
def find_missing_chars(k,s):
    my_string=s.replace("-","")
    seen=set(my_string)
    substring=""
    for i in range(1,3,len(s)):
        substring+=seen
    return substring

#30-second most repeated element in string
def second_most_repeted(s):
    cnt=Counter(s)
    most_common=cnt.most_common(2)
    if most_common<2:
        return ""
    else:
        return most_common[1][0]

#31-longest substring with at least k repeating charcter
def longestsubstring(s,k):
    for char,count in Counter(s).items():
        if count<k:
            return max(longest_substring(s,k) for s in s.split(char))
    return len(s)
#32-longest repeating substring
def longestrepeatingsubstring(s):
    dp=[[0]*len(s) for _ in range(len(s))]
    ans=0
    n=len(s)
    for i in range(n):
        for j in range(i+1,n):
            if s[i]==s[j]:
                dp[i][j]=dp[i-1][j-1]+1 if i else 1
                ans=max(ans,dp[i][j])
    return ans
#33-replace all "?" to avoid consequative repeating charcter
def modifystring(s):
    n=len(s)
    char_list=list(s)
    for i in range(n):
        if char_list[i]=="?":
            for x in "abc":
                if (i>0 and char_list[i-1]==c or i+1<n and char_list[i+1]==c):
                    countine
                char_list[i]=c
                break
    return "".join(char_list)
#34-longest increasing subsequence
def longest_invreasing_subsequence(nums):
    n=len(nums)
    dp=[1]*n
    for i in range(1,n):
        for j in range(i):
            if nums[i]>nums[j]:
                dp[i]=max(dp[i],dp[i+1])
    return max(dp)

#35-longest repeated substring
def longest_repeated_substring(s):
    n=len(s)
    dp=[[0]*n for _ in range(n)]
    ans=0
    for i in range(n):
        for j in range(1+1,n):
            if s[i]==s[j]:
                dp[i][j]=dp[i-1][j-1]+1 if i else 0
                ans=max(ans,dp[i][j])
    return ans

#36-max repeating substring
def max_repeating_subsequence(sequence,word):
    if len(subsequence)<len(word):return 0
    ans,k=0,1
    while word*k in sequence:
        ans+=1
        k+=1
    return ans

#37-count substring without repeating charcter
def numberOfSpecialSubstrings(s):
    count=0
    for i in range(len(s)):
        for j in range(i,len(s)):
            substring=s[i:j+1]
            if len(substring=len(set(substring)):
                count+=1
    return count
#38-longest repeating subsequence
def longest_repeating_subsequence(str):
    memo=[[0 for j in range(len(str)+1)] for i in range(len(str)+1)]
    for i in range(1,len(memo)):
        for j in range(1,len(memo[0])):
            if str[i-1]=str[j-1] and i!=j:
                memo[i][j]=memo[i-1][j-1]+1
            else:
                memo[i][j]=max(memo[i][j-1],memo[i-1][j])
    return memo[-1][-1]

#39-longest uncommon subsequence
def findluslength(a,b):
    if a==b: return -1
        return max(len(a),len(b))
#40-multiply substring
def multiply(nums1,nums2):
    n1,n2=0,0
    for i in nums1:
        n1=n1*10+(ord(i)-48)
    for i in nums2:
        n2=n2*10+(ord(i)-48)
    return n1*n2

#41-return all substring frm given string
def unique_substring(s):
    substring=set()
    for i in range(len(s)):
        for j in range(i+1,len(s)+1)
        substring.add(s[i:j])
    return list(substring)

#42-jewels and stone
def numjewelsandstones(jewels,stones):
    jeweles_set=set(jewels)
    return sum(stone in jewelset for stone in stones)

#43-ransome note
def cancostrcut(ransomenote,magazine):
    return (Counter(ransomenote)-Counter(magazine))=={}

#44-isisomorphic
def solution(x,y):
    return [s.index(t) for x in s]==[t.index(y) for y in t]

#45-rearrange space between words
def record_spaces(text):
    words=text.split()
    spaces=text.count(" ")
    if len(words)==1: 
        return words[0]+" "*spaces
    between,end=divmod(spaces,len(words)-1)
    return (' '*between).join(words)+" "*end

#46- rearrange charcter to make target string
def rearrange_charcter(s,target):
    s,target=Counter(s),Counter(target)
    return min(s[e]//target[e] for e in target)


#47-rearrange string so that no two adjcent charcter are same
def regonizestring(s):
    n,index,ans,char_list=len(s),0,[None]*n,Counter(s)
    if(max(char_list.values()>((n+1)//2))): return " "
    for char,val in char_list.most_common():
        while char_list:
            ans[index]=char
            v-=1
            index+=2
            if index>=n:
                index=1
    return "".join(ans)

#48-partition label
def partitions_label(s):
    last_occ={char:index for index,char in enumerate(s)}
    partitions,start,end=[],0,0
    for i,char in enumerate(s):
        last_index=last_occ[char]
        end=max(end,last_index)
        if i==end:
            partitions.append(i-start+1)
            start=i+1
    return partitions
#49- filp string to monotone increasing
def minfilpmonoincr(s):
    flip_count,one_count=0,0
    for i in range(len(s)):
        if s[i]=="1":
            one_count+=1
        else:
            flip_count=min(flip_count+1,one_count)
    return flip_count
#50-check balanced string
def isbalanced(num):
    odd,even=0,0
    for i in range(len(nums)):
        digit=int(num[i])
        if i%2==0:
            even+=digit
        else:
            odd+=digit
    return even==odd



##############################################################################################################################
############################################################ palindrome ######################################################
##############################################################################################################################
#51-vaild palindrome
def ispalindrome(s):
    s=s.lower()
    s="".join(filter(str.isalnum(),s))
    return 1 if s-s[::-1] else 0

#52-return true if string can be palindrome after delete at most one charcter
def vaild_palindrome2(s):
    if s==s[::-1]: return True
    l,r=0,len(s)-1
    while l<r:
        if s[l]!=s[r]:
            temp2=s[:r]+s[r+l:]
            temp=s[:l]+s[l+1:]
            return temp==temp[::-1] or temp2==temp2[::-1]
        l+=1
        r-=1
#53-palindrome operations
def solution(s):
    l,r=0,len(s)-1
    cnt=0
    while l<r:
        cnt+=s[i]!=s[j]
        i,j=i+1,i-1
    return cnt<=2

#54-unique length 3 palindrome substring
from itertools import ascii_lowercase
def countuniquepalindrome(s):
    count=0
    for char in ascii_lowercase:
        left_index=s.find(char)
        rigth_index=s.rfind(char)
        if rigth_index-left_index>1:
            count+=len(set(s[left_index+1:right_index]))
    return count
#55-convert to plaindrome
def slove(s):
    l,r=0,len(s)-1
    while l<r:
        if s[l]==s[r]:
            l+=1
            r-=1
        else:
            if s[l+1:r+1]==a[l+1:r+1][::-1]:
                return 1
            if s[l:r]==s[l:r][::-1]:
                return 1
            return 0
    return 1

#56-can permautions make string palindrome
def cnpermutepalindrome(s):
    return sum(v%2 for v in Counter(s).values())<=1










































































