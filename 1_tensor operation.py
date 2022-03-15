# Tensor
# - Creation (OBJ to Tensor, Types)
# - Shape (check, reshape, view)
# - Device change (cpu, CUDA)
# - Data Fetch (Tensor to OBJ)
# - 1 to N (a Tensor -> a list of Tensors)
# - N to 1 (a list of Tensors -> a Tensor)
# - Dimension Operation (squeeze, unsqueeze, permutation, transpose, ...)
# - Indexing
# - Reduce Operation(min, max, mean, argmax, argmin)
#----------------------------------------------------------------

#! Tensor Creation (OBJ to Tensor, Types)
import torch

#From scalar(float, int),
x = 34.0
y = torch.tensor(x) 
type(x), type(y) #(<class 'float'>, <class 'torch.Tensor'>)
x, y #(34.0, tensor(34.))

#From array(list,..)
x = [1,2,3,4,5]
y = torch.tensor(x) 
type(x), type(y) #(<class 'float'>, <class 'torch.Tensor'>)
x, y #([1, 2, 3, 4, 5], tensor([1, 2, 3, 4, 5]))
#Tensor : 스칼라, 벡터, 행렬, N차원 data 모두 텐서임

#From Matrix
x = [[1,2,3,4],
     [5,6,7,8]]
y = torch.tensor(x)
y # tensor([[1, 2, 3, 4],
#         [5, 6, 7, 8]])
y.shape #torch.Size([2, 4])
y.size() #torch.Size([2, 4])

#----------------------------------------------------------------
#* 보통 Numpy를 Tensor로 바꾸는 경우가 많다.
import numpy as np
a = 34.34
b = torch.tensor(a)
b, b.dtype #(tensor(34.3400), torch.float32)

#스칼라
x = np.array(34.3)
type(x)
x.dtype #dtype('float64')
y = torch.tensor(x)
y #tensor(34.3000, dtype=torch.float64)
#Numpy가 인식하는 데이터의 타입이 Tensor의 데이터 타입이 된다.

#행렬
x = np.array([[1,2,3,4],
            [5,6,7,8]])
x.shape #torch.Size([2, 4])
torch.tensor(x)
torch.tensor(x).shape #torch.Size([2, 4])

#N차원 데이터
x = np.array([[[[1,2,3,4],
            [5,6,7,8]]]])
x.shape #(1, 1, 2, 4)
torch.tensor(x)
torch.tensor(x).shape #torch.Size([1, 1, 2, 4])
#------------------------------------------------
#* Tensor dType
#float, integer, complex, boolean 
#상세한 것은 크기에 따라 나눠짐

#Type Conversion
x = torch.tensor([1,2,3,4], dtype=torch.float)
x, x.dtype #(tensor([1., 2., 3., 4.]), torch.float32)

x = torch.tensor([1,2,3,4], dtype=torch.float16)
x, x.dtype #(tensor([1., 2., 3., 4.], dtype=torch.float16), torch.float16)

#to()
x = torch.tensor([1,2,3,4], dtype=torch.float)
x, x.dtype #(tensor([1., 2., 3., 4.]), torch.float32)
y = x.to(torch.int)
y, y.dtype #(tensor([1, 2, 3, 4], dtype=torch.int32), torch.int32)

#type()
x = torch.tensor([1,2,3,4], dtype=torch.float)
x, x.dtype #(tensor([1., 2., 3., 4.]), torch.float32)
y = x.type(torch.int)
y, y.dtype #(tensor([1, 2, 3, 4], dtype=torch.int32), torch.int32)

#변경할 타입을 문자열로 지정
x = torch.tensor([1,2,3,4], dtype=torch.float)
x, x.dtype #(tensor([1., 2., 3., 4.]), torch.float32)
OUT_TYPE = 'torch.ShortTensor' 
y = x.type(OUT_TYPE) #문자열 해석해서 동적으로 알맞게 할당
y, y.dtype #(tensor([1, 2, 3, 4], dtype=torch.int16), torch.int16)
#공식홈페이지 : https://pytorch.org/docs/stable/tensor_attributes.html?highlight=dtype#torch.torch.dtype
#--------------------------------------------------------
#! Shape (check, reshape, view)
x = torch.tensor([[1,2,3],
                [4,5,6]])
x, x.shape, x.size()
batch_size, dim_1 = x.shape
batch_size, dim_1

#* Reshape() : 데이터의 전체 양이 같다면 차원 마음대로 변경가능
print("Original: ", x.shape)
print("Reshaped: ", x.reshape([3,2]).shape)
print("Reshaped: ", x.reshape([1,6]).shape)
print("Reshaped: ", x.reshape([6,1]).shape)
print("Reshaped: ", x.reshape([1,1,1,1,1,1,1,6]).shape)

x.reshape([3,2]) 
#tensor([[1, 2],
        # [3, 4],
        # [5, 6]])
x.reshape([1,6]) #tensor([[1, 2, 3, 4, 5, 6]])

#* view() : reshape()와 기능이 거의 같으나 메모리 사용이 다르다.
#*          데이터 자체의 메모리 구조를 바꾸지 않고, 메모리 접근할때 Shape만 바꿔준다.
#*          즉, 데이터는 바꾸지 않고 구조만 바뀌는 듯이 보임
x = torch.randn(4,4)
x.size()
y = x.view(16)
y.size()
z = x.view(-1, 8) #-1 사이즈는 명시적인 데이터로 추측한다.
#                 #총 16개의 데이터이고, 두번째 차원은 8사이즈로 고정하고 싶고 
#                  첫차원의 사이즈는 알아서 결정
z.size() #torch.Size([2, 8])
#-------------
a = torch.tensor([
    [
        [1,2,3,4],
        [5,6,7,8]
    ],[
        [11,12,13,14],
        [15,16,17,18]
    ],[
        [21,22,23,24],
        [25,26,27,28]
    ],
])
a , a.size()
b = a.view(3,4,2)
b
# tensor([[[ 1,  2],
#          [ 3,  4],
#          [ 5,  6],
#          [ 7,  8]],

#         [[11, 12],
#          [13, 14],
#          [15, 16],
#          [17, 18]],

#         [[21, 22],
#          [23, 24],
#          [25, 26],
#          [27, 28]]])

a[1] = torch.tensor([[51,52,53,54],
                     [55,56,57,58]])
b #b의 값이 바뀜

b[1] = torch.tensor([[151, 152],
                    [153, 154],
                    [155, 156],
                    [157, 158]])

a #a의 값도 바뀜

#reshape() 하더라도 a에 영향을 줌
c = a.reshape(3,4,2)
c[1] = torch.tensor([[11, 12],
                    [13, 14],
                    [15, 16],
                    [17, 18]])
#------------------------------------------------
#! Device change (cpu, CUDA)
#CPU To GPU
#GPU To cpu

a = torch.tensor([1,2,3,4], device='cpu') #디폴트
a, a.device #(tensor([1, 2, 3, 4]), device(type='cpu'))

a = torch.tensor([1,2,3,4], device='cuda')
a, a.device #(tensor([1, 2, 3, 4], device='cuda:0'), device(type='cuda', index=0))

a = torch.tensor([1,2,3,4], device='cuda:0') #그래픽 카드 번호(어떤 GPU에 할당할 것인지 결정)
a, a.device #(tensor([1, 2, 3, 4], device='cuda:0'), device(type='cuda', index=0))

#* CPU to GPU
a = torch.tensor([1,2,3,4])
b = a.to('cuda')
b.device
b[0] = 9
a,b

#* GPU to CPU
a = torch.tensor([
    [
        [1,2,3,4],
        [5,6,7,8]
    ],[
        [11,12,13,14],
        [15,16,17,18]
    ],[
        [21,22,23,24],
        [25,26,27,28]
    ],
]).to('cuda')
a, a.device

b = a.cpu()
b, b.device, type(b)

#! Data Fetch (Tensor to OBJ)
c = b.tolist()
c, type(c)

c = b.numpy()
c, type(c)

# a.numpy() #에러, CPU로 메모리 영역을 바꿔야 다른 객체로 변환 가능
a.cpu().numpy()
a.to('cpu').numpy()

#item() 스칼라값 CPU로 바로 얻을 때 사용
a = torch.tensor([1,2,3,4], device='cuda')
#a.item() #에러
a = torch.tensor([1], device='cuda')
b = a.item() 
b, type(b) #(1, <class 'int'>), CPU로 바꿀 필요 없음(자동)

a = torch.tensor([3.14], device='cuda')
b = a.item() 
b, type(b) #(3.140000104904175, <class 'float'>)

#! 1 to N (a Tensor -> a list of Tensors)
#chunk()
a = torch.tensor([
    [
        [1,2,3,4],
        [5,6,7,8]
    ],[
        [11,12,13,14],
        [15,16,17,18]
    ],[
        [21,22,23,24],
        [25,26,27,28]
    ],
])
a.shape #torch.Size([3, 2, 4])

a_l_tensors = torch.chunk(a, chunks=3, dim=0) #chunks : 몇개로 쪼갤지, dim: 어느 차원에서 쪼갤것인지
a_l_tensors #튜플
a_l_tensors[0] 
a_l_tensors[0].shape #torch.Size([1, 2, 4])

torch.chunk(a, chunks=4, dim=-1) #dim=-1은 마지막 차원
torch.chunk(a, chunks=4, dim=-1)[0] 
torch.chunk(a, chunks=4, dim=-1)[0].shape #torch.Size([3, 2, 1])


#! N to 1 (a list of Tensors -> a Tensor)
#cat(), stack()

#*cat()
a_l_tensors = torch.chunk(a, chunks=3, dim=0)
a_l_tensors
a_l_tensors[0]
a_l_tensors[0].shape #torch.Size([1, 2, 4])
a_l_tensors[1].shape #torch.Size([1, 2, 4])
a_l_tensors[2].shape #torch.Size([1, 2, 4])

b = torch.cat(a_l_tensors)
b
b.shape #torch.Size([3, 2, 4])

b = torch.cat(a_l_tensors, dim=1) #dim 디폴트 0
b
b.shape #torch.Size([1, 6, 4])

b = torch.cat(a_l_tensors, dim=-1) #dim =2와 같음
b
b.shape #torch.Size([1, 2, 12])

#* stack() : 새로운 차원에 따라 합친다.
a.shape #torch.Size([3, 2, 4])
a_l_tensors
e = torch.stack(a_l_tensors)
e
e.shape #torch.Size([3, 1, 2, 4]) #더미 차원 생성

#----------------------------------------
#! Dimension Operation (squeeze, unsqueeze, permutation, transpose, ...)
#transpose() : 차원을 서로 바꿈
a = torch.tensor([
    [
        [1,2,3,4],
        [5,6,7,8]
    ],[
        [11,12,13,14],
        [15,16,17,18]
    ],[
        [21,22,23,24],
        [25,26,27,28]
    ],
])
a.shape #torch.Size([3, 2, 4])

b = torch.transpose(a, 0, 1) #0차원과 1차원을 바꾼다
b
b.shape #torch.Size([2, 3, 4])


b = torch.transpose(a, 0, 2) #0차원과 마지막 차원을 바꾼다
b
b.shape #torch.Size([4, 2, 3])

#* permute() : 모든 차원을 한꺼번에 변경
a.shape #torch.Size([3, 2, 4])
d = a.permute(2,0,1) #4,3,2로 바꿔라
d.shape
d

#view() VS reshape(), view() VS trasfose(), transpose() VS permute()어떤것을 써야할지 아래 참조
#https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/
#transpose()로 변경시 인접한 데이터들간의 연속성이 사리진다.
#view()는 메모리에 연속된 데이터만 사용할 수 있다. 그리고 리턴되는 결과도 연속성을 유지
#b.view(8,3) 오류

#* squeeze() : 비어있는 차원 줄이기
x = torch.zeros(2,1,2,1,2)
x
y = torch.squeeze(x)
y, y.shape
# (tensor([[[0., 0.],
#          [0., 0.]],

#         [[0., 0.],
#          [0., 0.]]]), torch.Size([2, 2, 2]))

y = torch.squeeze(x, dim=1) #특정 차원만 줄이기
y.shape #torch.Size([2, 2, 1, 2]))

#* unsqueeze() : 비어있는 차원 만들기
y = torch.squeeze(x)
y.shape # torch.Size([2, 2, 2])
z = y.unsqueeze(dim=1)
z.shape #torch.Size([2, 1, 2, 2])
z

#! Indexing
#https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
x = [ [ 0,1,2,3,4,5,6,7,8,9 ], 
      [ 10,11,12,13,14,15,16,17,18,19 ], 
      [ 20,21,22,23,24,25,26,27,28,29 ], 
      [ 30,31,32,33,34,35,36,37,38,39 ], 
    ]
x = torch.tensor(x)
x       
print(x.shape) #torch.Size([4, 10])
x[:, 3] #tensor([ 3, 13, 23, 33])
x[3,:] #tensor([30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
x[:, 2:5]
x[:, -1]


indices = torch.LongTensor([3,7,1,4])
x[:,[3,7,1,4]]
x[:,indices]

#! Reduce Operation(min, max, mean, argmax, argmin)
a = torch.tensor( [
                        [
                            [ 1, 2, 3, 4 ], 
                            [ 5, 6, 7, 8 ], 
                        ],
                        [
                            [ 21, 22, 23, 24 ], 
                            [ 25, 26, 27, 28 ], 
                        ],
                        [
                            [ 11, 12, 13, 14 ], 
                            [ 15, 16, 17, 18 ], 
                        ],
                        
                  ]
                  )
print(a)
print(a.shape) #torch.Size([3, 2, 4])

torch.max(a) #전체에서 최댓값 : tensor(28) 
torch.max(a, dim=0) #첫번째 차원에서 max값 리턴
torch.max(a, dim=1) #두번째 차원에서 max값 리턴
torch.max(a, dim=-1) #마지막 차원에서 max값 리턴

torch.argmax(a, dim=0) #최댓값을 갖는 인덱스만
torch.argmax(a, dim=1)
torch.argmax(a, dim=2)

a.dtype 
b = a.to(torch.float)
torch.mean(b)
torch.mean(b, dim=(0,1,2))
torch.mean(b, dim=0)
torch.mean(b, dim=1)
torch.mean(b, dim=2)



