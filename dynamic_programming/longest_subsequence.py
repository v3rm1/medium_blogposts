def longest_common_subsequence(S1, p, S2, q):
    lcs_string = []
    if (p == 0) or (q == 0):
        return 0
    elif(S1[p] == S2[q]):
        lcs_string.append(S1[p])
        subseq = 1 + longest_common_subsequence(S1, p-1, S2, q-1)
    else:
        subseq = max(longest_common_subsequence(S1, p-1, S2, q), longest_common_subsequence(S1, p, S2, q-1))
    print(f"Longest Common Sub-sequence of {S1} and {S2}:\n{lcs_string}")
    print(f"Length of sub-sequence: {subseq}")
    return

def lcs_bottom_up(S1, p, S2, q):
    subseq = [[0 for x in range(q+1) for x in range(p+1)]]
    for i in range(p+1):
        for j in range(q+1):
            if (i==0) or (j==0):
                subseq[i][j] = 0
            elif S1[i-1] == S2[j-1]:
                subseq[i][j] = subseq[i-1][j-1]
            else:
                subseq[i][j] = max(subseq[i-1][j], subseq[i][j-1])
            
    
    index = subseq[p][q]
    lcs_string = [""] * (index+1)
    i = p
    j = q
    while i>0 and j>0:
        if S1[i-1] == S2[j-1]:
            lcs_string[index-1] = S1[i-1]
            i -= 1
            j -= 1
            index -= 1
        elif subseq[i-1][j] > subseq[i][j-1]:
            i -= i
        else:
            j -= 1
    print(f"Longest Common Sub-sequence of {S1} and {S2}:\n{lcs_string}")
    print(f"Length of sub-sequence: {max(subseq)}")
    return


S1 = "AGGTAB"
S2 = "GXTXAYB"
p = len(S1)-1
q = len(S2)-1
# longest_common_subsequence(S1, p, S2, q)
lcs_bottom_up(S1, p, S2, q)

