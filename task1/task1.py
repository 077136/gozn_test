
def multiplicate(a: list) -> list:
    def prod(a: list) -> tuple:
        elementProduct, numNulls, ind = 1, 0, None
        for i in a:     
            if i==0: 
                numNulls += 1
                ind = a.index(i)
            else: elementProduct *= i
        return (elementProduct, numNulls, ind)
    assert len(a)>1,  'Incorrect list, impossible compute output'
    elementProduct, numNulls, ind = prod(a)
    if numNulls > 1: b = [0]*len(a)
    elif numNulls == 1: 
        b = [0]*ind + [elementProduct] + [0]*(len(a)-ind-1)
    else: b = [elementProduct/i  for i in a]
    return b
    
if __name__ == "__main__":
    a = [1,2,3,4,5]
    print(multiplicate(a))