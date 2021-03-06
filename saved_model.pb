??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_40/kernel
}
$conv2d_40/kernel/Read/ReadVariableOpReadVariableOpconv2d_40/kernel*&
_output_shapes
:*
dtype0
t
conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_40/bias
m
"conv2d_40/bias/Read/ReadVariableOpReadVariableOpconv2d_40/bias*
_output_shapes
:*
dtype0
?
batch_normalization_32/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_32/gamma
?
0batch_normalization_32/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_32/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_32/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_32/beta
?
/batch_normalization_32/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_32/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_32/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_32/moving_mean
?
6batch_normalization_32/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_32/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_32/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_32/moving_variance
?
:batch_normalization_32/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_32/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameconv2d_41/kernel
~
$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*'
_output_shapes
:?*
dtype0
u
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_41/bias
n
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_33/gamma
?
0batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_33/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_33/beta
?
/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_33/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_33/moving_mean
?
6batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_33/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_33/moving_variance
?
:batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_33/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_42/kernel

$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_42/bias
n
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes	
:?*
dtype0
?
conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_43/kernel

$conv2d_43/kernel/Read/ReadVariableOpReadVariableOpconv2d_43/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_43/bias
n
"conv2d_43/bias/Read/ReadVariableOpReadVariableOpconv2d_43/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_34/gamma
?
0batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_34/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_34/beta
?
/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_34/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_34/moving_mean
?
6batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_34/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_34/moving_variance
?
:batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_34/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_44/kernel

$conv2d_44/kernel/Read/ReadVariableOpReadVariableOpconv2d_44/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_44/bias
n
"conv2d_44/bias/Read/ReadVariableOpReadVariableOpconv2d_44/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_35/gamma
?
0batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_35/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_35/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_35/beta
?
/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_35/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_35/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_35/moving_mean
?
6batch_normalization_35/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_35/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_35/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_35/moving_variance
?
:batch_normalization_35/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_35/moving_variance*
_output_shapes	
:?*
dtype0
{
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	
* 
shared_namedense_16/kernel
t
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes
:	?	
*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:
*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:
*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_40/kernel/m
?
+Adam/conv2d_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_40/bias/m
{
)Adam/conv2d_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_32/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_32/gamma/m
?
7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_32/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_32/beta/m
?
6Adam/batch_normalization_32/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv2d_41/kernel/m
?
+Adam/conv2d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/m*'
_output_shapes
:?*
dtype0
?
Adam/conv2d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_41/bias/m
|
)Adam/conv2d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_33/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_33/gamma/m
?
7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_33/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_33/beta/m
?
6Adam/batch_normalization_33/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_42/kernel/m
?
+Adam/conv2d_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_42/bias/m
|
)Adam/conv2d_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_43/kernel/m
?
+Adam/conv2d_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_43/bias/m
|
)Adam/conv2d_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_34/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_34/gamma/m
?
7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_34/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_34/beta/m
?
6Adam/batch_normalization_34/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_44/kernel/m
?
+Adam/conv2d_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_44/bias/m
|
)Adam/conv2d_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_35/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_35/gamma/m
?
7Adam/batch_normalization_35/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_35/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_35/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_35/beta/m
?
6Adam/batch_normalization_35/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_35/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	
*'
shared_nameAdam/dense_16/kernel/m
?
*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes
:	?	
*
dtype0
?
Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_16/bias/m
y
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_17/kernel/m
?
*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/m
y
(Adam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_40/kernel/v
?
+Adam/conv2d_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_40/bias/v
{
)Adam/conv2d_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_32/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_32/gamma/v
?
7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_32/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_32/beta/v
?
6Adam/batch_normalization_32/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv2d_41/kernel/v
?
+Adam/conv2d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/v*'
_output_shapes
:?*
dtype0
?
Adam/conv2d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_41/bias/v
|
)Adam/conv2d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_33/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_33/gamma/v
?
7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_33/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_33/beta/v
?
6Adam/batch_normalization_33/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_42/kernel/v
?
+Adam/conv2d_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_42/bias/v
|
)Adam/conv2d_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_43/kernel/v
?
+Adam/conv2d_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_43/bias/v
|
)Adam/conv2d_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_34/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_34/gamma/v
?
7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_34/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_34/beta/v
?
6Adam/batch_normalization_34/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_44/kernel/v
?
+Adam/conv2d_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_44/bias/v
|
)Adam/conv2d_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_35/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_35/gamma/v
?
7Adam/batch_normalization_35/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_35/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_35/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_35/beta/v
?
6Adam/batch_normalization_35/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_35/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	
*'
shared_nameAdam/dense_16/kernel/v
?
*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes
:	?	
*
dtype0
?
Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_16/bias/v
y
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_17/kernel/v
?
*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/v
y
(Adam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer-20
layer_with_weights-10
layer-21
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
R
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0trainable_variables
1	variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
R
:trainable_variables
;	variables
<regularization_losses
=	keras_api
R
>trainable_variables
?	variables
@regularization_losses
A	keras_api
?
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
R
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
R
[trainable_variables
\	variables
]regularization_losses
^	keras_api
R
_trainable_variables
`	variables
aregularization_losses
b	keras_api
?
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
htrainable_variables
i	variables
jregularization_losses
k	keras_api
h

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
R
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
R
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
?
zaxis
	{gamma
|beta
}moving_mean
~moving_variance
trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?,m?-m?4m?5m?Cm?Dm?Km?Lm?Um?Vm?dm?em?lm?mm?{m?|m?	?m?	?m?	?m?	?m?v?v?,v?-v?4v?5v?Cv?Dv?Kv?Lv?Uv?Vv?dv?ev?lv?mv?{v?|v?	?v?	?v?	?v?	?v?
?
0
1
,2
-3
44
55
C6
D7
K8
L9
U10
V11
d12
e13
l14
m15
{16
|17
?18
?19
?20
?21
?
0
1
,2
-3
.4
/5
46
57
C8
D9
E10
F11
K12
L13
U14
V15
d16
e17
f18
g19
l20
m21
{22
|23
}24
~25
?26
?27
?28
?29
 
?
trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
	variables
 ?layer_regularization_losses
regularization_losses
?layers
 
\Z
VARIABLE_VALUEconv2d_40/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_40/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
 	variables
 ?layer_regularization_losses
!regularization_losses
?layers
 
 
 
?
#trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
$	variables
 ?layer_regularization_losses
%regularization_losses
?layers
 
 
 
?
'trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
(	variables
 ?layer_regularization_losses
)regularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_32/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_32/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_32/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_32/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
.2
/3
 
?
0trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
1	variables
 ?layer_regularization_losses
2regularization_losses
?layers
\Z
VARIABLE_VALUEconv2d_41/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_41/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
?
6trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
7	variables
 ?layer_regularization_losses
8regularization_losses
?layers
 
 
 
?
:trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
;	variables
 ?layer_regularization_losses
<regularization_losses
?layers
 
 
 
?
>trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
@regularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_33/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_33/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_33/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_33/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
E2
F3
 
?
Gtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
H	variables
 ?layer_regularization_losses
Iregularization_losses
?layers
\Z
VARIABLE_VALUEconv2d_42/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_42/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 
?
Mtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
N	variables
 ?layer_regularization_losses
Oregularization_losses
?layers
 
 
 
?
Qtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
R	variables
 ?layer_regularization_losses
Sregularization_losses
?layers
\Z
VARIABLE_VALUEconv2d_43/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_43/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
?
Wtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
X	variables
 ?layer_regularization_losses
Yregularization_losses
?layers
 
 
 
?
[trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
\	variables
 ?layer_regularization_losses
]regularization_losses
?layers
 
 
 
?
_trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
`	variables
 ?layer_regularization_losses
aregularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_34/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_34/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_34/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_34/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

d0
e1

d0
e1
f2
g3
 
?
htrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
i	variables
 ?layer_regularization_losses
jregularization_losses
?layers
\Z
VARIABLE_VALUEconv2d_44/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_44/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

l0
m1
 
?
ntrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
o	variables
 ?layer_regularization_losses
pregularization_losses
?layers
 
 
 
?
rtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
s	variables
 ?layer_regularization_losses
tregularization_losses
?layers
 
 
 
?
vtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
w	variables
 ?layer_regularization_losses
xregularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_35/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_35/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_35/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_35/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

{0
|1
}2
~3
 
?
trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
?
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
?
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
\Z
VARIABLE_VALUEdense_17/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_17/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
8
.0
/1
E2
F3
f4
g5
}6
~7
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

.0
/1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

E0
F1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

f0
g1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

}0
~1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_40/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_32/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_33/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_42/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_42/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_43/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_43/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_34/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_34/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_44/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_44/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_35/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_35/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_17/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_17/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_40/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_32/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_33/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_42/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_42/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_43/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_43/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_34/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_34/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_44/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_44/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_35/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_35/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_17/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_17/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_40_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_40_inputconv2d_40/kernelconv2d_40/biasbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_varianceconv2d_41/kernelconv2d_41/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_44/kernelconv2d_44/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variancedense_16/kerneldense_16/biasdense_17/kerneldense_17/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8? *,
f'R%
#__inference_signature_wrapper_24938
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
? 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_40/kernel/Read/ReadVariableOp"conv2d_40/bias/Read/ReadVariableOp0batch_normalization_32/gamma/Read/ReadVariableOp/batch_normalization_32/beta/Read/ReadVariableOp6batch_normalization_32/moving_mean/Read/ReadVariableOp:batch_normalization_32/moving_variance/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp0batch_normalization_33/gamma/Read/ReadVariableOp/batch_normalization_33/beta/Read/ReadVariableOp6batch_normalization_33/moving_mean/Read/ReadVariableOp:batch_normalization_33/moving_variance/Read/ReadVariableOp$conv2d_42/kernel/Read/ReadVariableOp"conv2d_42/bias/Read/ReadVariableOp$conv2d_43/kernel/Read/ReadVariableOp"conv2d_43/bias/Read/ReadVariableOp0batch_normalization_34/gamma/Read/ReadVariableOp/batch_normalization_34/beta/Read/ReadVariableOp6batch_normalization_34/moving_mean/Read/ReadVariableOp:batch_normalization_34/moving_variance/Read/ReadVariableOp$conv2d_44/kernel/Read/ReadVariableOp"conv2d_44/bias/Read/ReadVariableOp0batch_normalization_35/gamma/Read/ReadVariableOp/batch_normalization_35/beta/Read/ReadVariableOp6batch_normalization_35/moving_mean/Read/ReadVariableOp:batch_normalization_35/moving_variance/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_40/kernel/m/Read/ReadVariableOp)Adam/conv2d_40/bias/m/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_32/beta/m/Read/ReadVariableOp+Adam/conv2d_41/kernel/m/Read/ReadVariableOp)Adam/conv2d_41/bias/m/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_33/beta/m/Read/ReadVariableOp+Adam/conv2d_42/kernel/m/Read/ReadVariableOp)Adam/conv2d_42/bias/m/Read/ReadVariableOp+Adam/conv2d_43/kernel/m/Read/ReadVariableOp)Adam/conv2d_43/bias/m/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_34/beta/m/Read/ReadVariableOp+Adam/conv2d_44/kernel/m/Read/ReadVariableOp)Adam/conv2d_44/bias/m/Read/ReadVariableOp7Adam/batch_normalization_35/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_35/beta/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp(Adam/dense_17/bias/m/Read/ReadVariableOp+Adam/conv2d_40/kernel/v/Read/ReadVariableOp)Adam/conv2d_40/bias/v/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_32/beta/v/Read/ReadVariableOp+Adam/conv2d_41/kernel/v/Read/ReadVariableOp)Adam/conv2d_41/bias/v/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_33/beta/v/Read/ReadVariableOp+Adam/conv2d_42/kernel/v/Read/ReadVariableOp)Adam/conv2d_42/bias/v/Read/ReadVariableOp+Adam/conv2d_43/kernel/v/Read/ReadVariableOp)Adam/conv2d_43/bias/v/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_34/beta/v/Read/ReadVariableOp+Adam/conv2d_44/kernel/v/Read/ReadVariableOp)Adam/conv2d_44/bias/v/Read/ReadVariableOp7Adam/batch_normalization_35/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_35/beta/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp(Adam/dense_17/bias/v/Read/ReadVariableOpConst*`
TinY
W2U	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *'
f"R 
__inference__traced_save_26322
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_40/kernelconv2d_40/biasbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_varianceconv2d_41/kernelconv2d_41/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_44/kernelconv2d_44/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variancedense_16/kerneldense_16/biasdense_17/kerneldense_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_40/kernel/mAdam/conv2d_40/bias/m#Adam/batch_normalization_32/gamma/m"Adam/batch_normalization_32/beta/mAdam/conv2d_41/kernel/mAdam/conv2d_41/bias/m#Adam/batch_normalization_33/gamma/m"Adam/batch_normalization_33/beta/mAdam/conv2d_42/kernel/mAdam/conv2d_42/bias/mAdam/conv2d_43/kernel/mAdam/conv2d_43/bias/m#Adam/batch_normalization_34/gamma/m"Adam/batch_normalization_34/beta/mAdam/conv2d_44/kernel/mAdam/conv2d_44/bias/m#Adam/batch_normalization_35/gamma/m"Adam/batch_normalization_35/beta/mAdam/dense_16/kernel/mAdam/dense_16/bias/mAdam/dense_17/kernel/mAdam/dense_17/bias/mAdam/conv2d_40/kernel/vAdam/conv2d_40/bias/v#Adam/batch_normalization_32/gamma/v"Adam/batch_normalization_32/beta/vAdam/conv2d_41/kernel/vAdam/conv2d_41/bias/v#Adam/batch_normalization_33/gamma/v"Adam/batch_normalization_33/beta/vAdam/conv2d_42/kernel/vAdam/conv2d_42/bias/vAdam/conv2d_43/kernel/vAdam/conv2d_43/bias/v#Adam/batch_normalization_34/gamma/v"Adam/batch_normalization_34/beta/vAdam/conv2d_44/kernel/vAdam/conv2d_44/bias/v#Adam/batch_normalization_35/gamma/v"Adam/batch_normalization_35/beta/vAdam/dense_16/kernel/vAdam/dense_16/bias/vAdam/dense_17/kernel/vAdam/dense_17/bias/v*_
TinX
V2T*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? **
f%R#
!__inference__traced_restore_26581??
?
g
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_23538

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_8_layer_call_fn_26030

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_244332
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25382

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_25668

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_8_layer_call_and_return_conditional_losses_24381

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_23961

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????..:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????..2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????..::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????..
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_35_layer_call_fn_25908

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_238692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_41_layer_call_and_return_conditional_losses_24007

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????**?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????..::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????..
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25446

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????..:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????..2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????..::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????..
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_33_layer_call_fn_25616

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_236062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
I
-__inference_activation_42_layer_call_fn_25658

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_42_layer_call_and_return_conditional_losses_241412
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25789

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25771

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_33_layer_call_fn_23544

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_235382
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25725

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_26015

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
D__inference_conv2d_40_layer_call_and_return_conditional_losses_25325

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????\\2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_42_layer_call_fn_25648

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_241202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_23490

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
I
-__inference_activation_44_layer_call_fn_25844

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_44_layer_call_and_return_conditional_losses_242932
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
ޜ
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_25185

inputs,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource2
.batch_normalization_32_readvariableop_resource4
0batch_normalization_32_readvariableop_1_resourceC
?batch_normalization_32_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource2
.batch_normalization_33_readvariableop_resource4
0batch_normalization_33_readvariableop_1_resourceC
?batch_normalization_33_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_42_conv2d_readvariableop_resource-
)conv2d_42_biasadd_readvariableop_resource,
(conv2d_43_conv2d_readvariableop_resource-
)conv2d_43_biasadd_readvariableop_resource2
.batch_normalization_34_readvariableop_resource4
0batch_normalization_34_readvariableop_1_resourceC
?batch_normalization_34_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_44_conv2d_readvariableop_resource-
)conv2d_44_biasadd_readvariableop_resource2
.batch_normalization_35_readvariableop_resource4
0batch_normalization_35_readvariableop_1_resourceC
?batch_normalization_35_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity??6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_32/ReadVariableOp?'batch_normalization_32/ReadVariableOp_1?6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_33/ReadVariableOp?'batch_normalization_33/ReadVariableOp_1?6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_34/ReadVariableOp?'batch_normalization_34/ReadVariableOp_1?6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_35/ReadVariableOp?'batch_normalization_35/ReadVariableOp_1? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_40/Conv2D/ReadVariableOp?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\*
paddingVALID*
strides
2
conv2d_40/Conv2D?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\2
conv2d_40/BiasAdd?
activation_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\\2
activation_40/Relu?
max_pooling2d_32/MaxPoolMaxPool activation_40/Relu:activations:0*/
_output_shapes
:?????????..*
ksize
*
paddingVALID*
strides
2
max_pooling2d_32/MaxPool?
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_32/ReadVariableOp?
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_32/ReadVariableOp_1?
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_32/MaxPool:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????..:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_32/FusedBatchNormV3?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2DConv2D+batch_normalization_32/FusedBatchNormV3:y:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?*
paddingVALID*
strides
2
conv2d_41/Conv2D?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?2
conv2d_41/BiasAdd?
activation_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:?????????**?2
activation_41/Relu?
max_pooling2d_33/MaxPoolMaxPool activation_41/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_33/MaxPool?
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_33/ReadVariableOp?
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_33/ReadVariableOp_1?
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_33/MaxPool:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_33/FusedBatchNormV3?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2DConv2D+batch_normalization_33/FusedBatchNormV3:y:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_42/Conv2D?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_42/BiasAdd?
activation_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_42/Relu?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_43/Conv2D/ReadVariableOp?
conv2d_43/Conv2DConv2D activation_42/Relu:activations:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_43/Conv2D?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_43/BiasAdd/ReadVariableOp?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_43/BiasAdd?
activation_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_43/Relu?
max_pooling2d_34/MaxPoolMaxPool activation_43/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_34/MaxPool?
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_34/ReadVariableOp?
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_34/ReadVariableOp_1?
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_34/MaxPool:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_34/FusedBatchNormV3?
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_44/Conv2D/ReadVariableOp?
conv2d_44/Conv2DConv2D+batch_normalization_34/FusedBatchNormV3:y:0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_44/Conv2D?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_44/BiasAdd/ReadVariableOp?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_44/BiasAdd?
activation_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_44/Relu?
max_pooling2d_35/MaxPoolMaxPool activation_44/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_35/MaxPool?
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_35/ReadVariableOp?
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_35/ReadVariableOp_1?
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_35/MaxPool:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_35/FusedBatchNormV3s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_8/Const?
flatten_8/ReshapeReshape+batch_normalization_35/FusedBatchNormV3:y:0flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten_8/Reshape?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	?	
*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulflatten_8/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_16/Relu?
dropout_8/IdentityIdentitydense_16/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
dropout_8/Identity?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldropout_8/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Sigmoid?

IdentityIdentitydense_17/Sigmoid:y:07^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_8_layer_call_fn_25315

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_248002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
}
(__inference_dense_17_layer_call_fn_26050

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_244572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25539

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_44_layer_call_and_return_conditional_losses_24293

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_33_layer_call_fn_25552

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_240562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_23722

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_8_layer_call_fn_24863
conv2d_40_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_248002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_40_input
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_23869

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_32_layer_call_fn_25472

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_239612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????..2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????..::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????..
 
_user_specified_nameinputs
?
~
)__inference_conv2d_44_layer_call_fn_25834

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_242722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_8_layer_call_fn_25250

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_246492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_24428

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_23521

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_24074

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_24159

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_41_layer_call_and_return_conditional_losses_25496

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????**?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????**?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????**?:X T
0
_output_shapes
:?????????**?
 
_user_specified_nameinputs
?f
?

G__inference_sequential_8_layer_call_and_return_conditional_losses_24800

inputs
conv2d_40_24717
conv2d_40_24719 
batch_normalization_32_24724 
batch_normalization_32_24726 
batch_normalization_32_24728 
batch_normalization_32_24730
conv2d_41_24733
conv2d_41_24735 
batch_normalization_33_24740 
batch_normalization_33_24742 
batch_normalization_33_24744 
batch_normalization_33_24746
conv2d_42_24749
conv2d_42_24751
conv2d_43_24755
conv2d_43_24757 
batch_normalization_34_24762 
batch_normalization_34_24764 
batch_normalization_34_24766 
batch_normalization_34_24768
conv2d_44_24771
conv2d_44_24773 
batch_normalization_35_24778 
batch_normalization_35_24780 
batch_normalization_35_24782 
batch_normalization_35_24784
dense_16_24788
dense_16_24790
dense_17_24794
dense_17_24796
identity??.batch_normalization_32/StatefulPartitionedCall?.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_44/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_24717conv2d_40_24719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_238942#
!conv2d_40/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_239152
activation_40/PartitionedCall?
 max_pooling2d_32/PartitionedCallPartitionedCall&activation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_234222"
 max_pooling2d_32/PartitionedCall?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0batch_normalization_32_24724batch_normalization_32_24726batch_normalization_32_24728batch_normalization_32_24730*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_2396120
.batch_normalization_32/StatefulPartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_41_24733conv2d_41_24735*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_240072#
!conv2d_41/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_240282
activation_41/PartitionedCall?
 max_pooling2d_33/PartitionedCallPartitionedCall&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_235382"
 max_pooling2d_33/PartitionedCall?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_33_24740batch_normalization_33_24742batch_normalization_33_24744batch_normalization_33_24746*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2407420
.batch_normalization_33/StatefulPartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_42_24749conv2d_42_24751*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_241202#
!conv2d_42/StatefulPartitionedCall?
activation_42/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_42_layer_call_and_return_conditional_losses_241412
activation_42/PartitionedCall?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_43_24755conv2d_43_24757*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_241592#
!conv2d_43/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_241802
activation_43/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_236542"
 max_pooling2d_34/PartitionedCall?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_34_24762batch_normalization_34_24764batch_normalization_34_24766batch_normalization_34_24768*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2422620
.batch_normalization_34/StatefulPartitionedCall?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0conv2d_44_24771conv2d_44_24773*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_242722#
!conv2d_44/StatefulPartitionedCall?
activation_44/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_44_layer_call_and_return_conditional_losses_242932
activation_44/PartitionedCall?
 max_pooling2d_35/PartitionedCallPartitionedCall&activation_44/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_237702"
 max_pooling2d_35/PartitionedCall?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_35_24778batch_normalization_35_24780batch_normalization_35_24782batch_normalization_35_24784*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2433920
.batch_normalization_35/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_243812
flatten_8/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_16_24788dense_16_24790*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_244002"
 dense_16/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_244332
dropout_8/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_17_24794dense_17_24796*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_244572"
 dense_17/StatefulPartitionedCall?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25585

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_33_layer_call_fn_25629

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_236372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_42_layer_call_and_return_conditional_losses_24141

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_16_layer_call_and_return_conditional_losses_25994

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25603

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_24339

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_32_layer_call_fn_25408

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_235212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_dense_16_layer_call_fn_26003

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_244002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25364

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_34_layer_call_fn_25738

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_242082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_32_layer_call_fn_25459

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_239432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????..2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????..::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????..
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_23753

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25882

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_32_layer_call_fn_23428

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_234222
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?%
__inference__traced_save_26322
file_prefix/
+savev2_conv2d_40_kernel_read_readvariableop-
)savev2_conv2d_40_bias_read_readvariableop;
7savev2_batch_normalization_32_gamma_read_readvariableop:
6savev2_batch_normalization_32_beta_read_readvariableopA
=savev2_batch_normalization_32_moving_mean_read_readvariableopE
Asavev2_batch_normalization_32_moving_variance_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop;
7savev2_batch_normalization_33_gamma_read_readvariableop:
6savev2_batch_normalization_33_beta_read_readvariableopA
=savev2_batch_normalization_33_moving_mean_read_readvariableopE
Asavev2_batch_normalization_33_moving_variance_read_readvariableop/
+savev2_conv2d_42_kernel_read_readvariableop-
)savev2_conv2d_42_bias_read_readvariableop/
+savev2_conv2d_43_kernel_read_readvariableop-
)savev2_conv2d_43_bias_read_readvariableop;
7savev2_batch_normalization_34_gamma_read_readvariableop:
6savev2_batch_normalization_34_beta_read_readvariableopA
=savev2_batch_normalization_34_moving_mean_read_readvariableopE
Asavev2_batch_normalization_34_moving_variance_read_readvariableop/
+savev2_conv2d_44_kernel_read_readvariableop-
)savev2_conv2d_44_bias_read_readvariableop;
7savev2_batch_normalization_35_gamma_read_readvariableop:
6savev2_batch_normalization_35_beta_read_readvariableopA
=savev2_batch_normalization_35_moving_mean_read_readvariableopE
Asavev2_batch_normalization_35_moving_variance_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_40_kernel_m_read_readvariableop4
0savev2_adam_conv2d_40_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_m_read_readvariableop6
2savev2_adam_conv2d_41_kernel_m_read_readvariableop4
0savev2_adam_conv2d_41_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_m_read_readvariableop6
2savev2_adam_conv2d_42_kernel_m_read_readvariableop4
0savev2_adam_conv2d_42_bias_m_read_readvariableop6
2savev2_adam_conv2d_43_kernel_m_read_readvariableop4
0savev2_adam_conv2d_43_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_m_read_readvariableop6
2savev2_adam_conv2d_44_kernel_m_read_readvariableop4
0savev2_adam_conv2d_44_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_35_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_35_beta_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop3
/savev2_adam_dense_17_bias_m_read_readvariableop6
2savev2_adam_conv2d_40_kernel_v_read_readvariableop4
0savev2_adam_conv2d_40_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_v_read_readvariableop6
2savev2_adam_conv2d_41_kernel_v_read_readvariableop4
0savev2_adam_conv2d_41_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_v_read_readvariableop6
2savev2_adam_conv2d_42_kernel_v_read_readvariableop4
0savev2_adam_conv2d_42_bias_v_read_readvariableop6
2savev2_adam_conv2d_43_kernel_v_read_readvariableop4
0savev2_adam_conv2d_43_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_v_read_readvariableop6
2savev2_adam_conv2d_44_kernel_v_read_readvariableop4
0savev2_adam_conv2d_44_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_35_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_35_beta_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop3
/savev2_adam_dense_17_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?.
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*?-
value?-B?-TB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_40_kernel_read_readvariableop)savev2_conv2d_40_bias_read_readvariableop7savev2_batch_normalization_32_gamma_read_readvariableop6savev2_batch_normalization_32_beta_read_readvariableop=savev2_batch_normalization_32_moving_mean_read_readvariableopAsavev2_batch_normalization_32_moving_variance_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop7savev2_batch_normalization_33_gamma_read_readvariableop6savev2_batch_normalization_33_beta_read_readvariableop=savev2_batch_normalization_33_moving_mean_read_readvariableopAsavev2_batch_normalization_33_moving_variance_read_readvariableop+savev2_conv2d_42_kernel_read_readvariableop)savev2_conv2d_42_bias_read_readvariableop+savev2_conv2d_43_kernel_read_readvariableop)savev2_conv2d_43_bias_read_readvariableop7savev2_batch_normalization_34_gamma_read_readvariableop6savev2_batch_normalization_34_beta_read_readvariableop=savev2_batch_normalization_34_moving_mean_read_readvariableopAsavev2_batch_normalization_34_moving_variance_read_readvariableop+savev2_conv2d_44_kernel_read_readvariableop)savev2_conv2d_44_bias_read_readvariableop7savev2_batch_normalization_35_gamma_read_readvariableop6savev2_batch_normalization_35_beta_read_readvariableop=savev2_batch_normalization_35_moving_mean_read_readvariableopAsavev2_batch_normalization_35_moving_variance_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_40_kernel_m_read_readvariableop0savev2_adam_conv2d_40_bias_m_read_readvariableop>savev2_adam_batch_normalization_32_gamma_m_read_readvariableop=savev2_adam_batch_normalization_32_beta_m_read_readvariableop2savev2_adam_conv2d_41_kernel_m_read_readvariableop0savev2_adam_conv2d_41_bias_m_read_readvariableop>savev2_adam_batch_normalization_33_gamma_m_read_readvariableop=savev2_adam_batch_normalization_33_beta_m_read_readvariableop2savev2_adam_conv2d_42_kernel_m_read_readvariableop0savev2_adam_conv2d_42_bias_m_read_readvariableop2savev2_adam_conv2d_43_kernel_m_read_readvariableop0savev2_adam_conv2d_43_bias_m_read_readvariableop>savev2_adam_batch_normalization_34_gamma_m_read_readvariableop=savev2_adam_batch_normalization_34_beta_m_read_readvariableop2savev2_adam_conv2d_44_kernel_m_read_readvariableop0savev2_adam_conv2d_44_bias_m_read_readvariableop>savev2_adam_batch_normalization_35_gamma_m_read_readvariableop=savev2_adam_batch_normalization_35_beta_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop/savev2_adam_dense_17_bias_m_read_readvariableop2savev2_adam_conv2d_40_kernel_v_read_readvariableop0savev2_adam_conv2d_40_bias_v_read_readvariableop>savev2_adam_batch_normalization_32_gamma_v_read_readvariableop=savev2_adam_batch_normalization_32_beta_v_read_readvariableop2savev2_adam_conv2d_41_kernel_v_read_readvariableop0savev2_adam_conv2d_41_bias_v_read_readvariableop>savev2_adam_batch_normalization_33_gamma_v_read_readvariableop=savev2_adam_batch_normalization_33_beta_v_read_readvariableop2savev2_adam_conv2d_42_kernel_v_read_readvariableop0savev2_adam_conv2d_42_bias_v_read_readvariableop2savev2_adam_conv2d_43_kernel_v_read_readvariableop0savev2_adam_conv2d_43_bias_v_read_readvariableop>savev2_adam_batch_normalization_34_gamma_v_read_readvariableop=savev2_adam_batch_normalization_34_beta_v_read_readvariableop2savev2_adam_conv2d_44_kernel_v_read_readvariableop0savev2_adam_conv2d_44_bias_v_read_readvariableop>savev2_adam_batch_normalization_35_gamma_v_read_readvariableop=savev2_adam_batch_normalization_35_beta_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop/savev2_adam_dense_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *b
dtypesX
V2T	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::?:?:?:?:?:?:??:?:??:?:?:?:?:?:??:?:?:?:?:?:	?	
:
:
:: : : : : : : : : :::::?:?:?:?:??:?:??:?:?:?:??:?:?:?:	?	
:
:
::::::?:?:?:?:??:?:??:?:?:?:??:?:?:?:	?	
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :,((
&
_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
::-,)
'
_output_shapes
:?:!-

_output_shapes	
:?:!.

_output_shapes	
:?:!/

_output_shapes	
:?:.0*
(
_output_shapes
:??:!1

_output_shapes	
:?:.2*
(
_output_shapes
:??:!3

_output_shapes	
:?:!4

_output_shapes	
:?:!5

_output_shapes	
:?:.6*
(
_output_shapes
:??:!7

_output_shapes	
:?:!8

_output_shapes	
:?:!9

_output_shapes	
:?:%:!

_output_shapes
:	?	
: ;

_output_shapes
:
:$< 

_output_shapes

:
: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
:: A

_output_shapes
::-B)
'
_output_shapes
:?:!C

_output_shapes	
:?:!D

_output_shapes	
:?:!E

_output_shapes	
:?:.F*
(
_output_shapes
:??:!G

_output_shapes	
:?:.H*
(
_output_shapes
:??:!I

_output_shapes	
:?:!J

_output_shapes	
:?:!K

_output_shapes	
:?:.L*
(
_output_shapes
:??:!M

_output_shapes	
:?:!N

_output_shapes	
:?:!O

_output_shapes	
:?:%P!

_output_shapes
:	?	
: Q

_output_shapes
:
:$R 

_output_shapes

:
: S

_output_shapes
::T

_output_shapes
: 
?
?
#__inference_signature_wrapper_24938
conv2d_40_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8? *)
f$R"
 __inference__wrapped_model_234162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_40_input
?
?
6__inference_batch_normalization_33_layer_call_fn_25565

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_240742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_42_layer_call_and_return_conditional_losses_25639

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_24433

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????
2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????
2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
D__inference_conv2d_42_layer_call_and_return_conditional_losses_24120

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_23422

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_8_layer_call_fn_26025

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_244282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????
22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
D__inference_conv2d_41_layer_call_and_return_conditional_losses_25482

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????**?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????..::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????..
 
_user_specified_nameinputs
?
~
)__inference_conv2d_41_layer_call_fn_25491

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_240072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????**?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????..::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????..
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_23606

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_17_layer_call_and_return_conditional_losses_26041

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
d
H__inference_activation_40_layer_call_and_return_conditional_losses_23915

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????\\2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????\\2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????\\:W S
/
_output_shapes
:?????????\\
 
_user_specified_nameinputs
?h
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_24474
conv2d_40_input
conv2d_40_23905
conv2d_40_23907 
batch_normalization_32_23988 
batch_normalization_32_23990 
batch_normalization_32_23992 
batch_normalization_32_23994
conv2d_41_24018
conv2d_41_24020 
batch_normalization_33_24101 
batch_normalization_33_24103 
batch_normalization_33_24105 
batch_normalization_33_24107
conv2d_42_24131
conv2d_42_24133
conv2d_43_24170
conv2d_43_24172 
batch_normalization_34_24253 
batch_normalization_34_24255 
batch_normalization_34_24257 
batch_normalization_34_24259
conv2d_44_24283
conv2d_44_24285 
batch_normalization_35_24366 
batch_normalization_35_24368 
batch_normalization_35_24370 
batch_normalization_35_24372
dense_16_24411
dense_16_24413
dense_17_24468
dense_17_24470
identity??.batch_normalization_32/StatefulPartitionedCall?.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_44/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputconv2d_40_23905conv2d_40_23907*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_238942#
!conv2d_40/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_239152
activation_40/PartitionedCall?
 max_pooling2d_32/PartitionedCallPartitionedCall&activation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_234222"
 max_pooling2d_32/PartitionedCall?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0batch_normalization_32_23988batch_normalization_32_23990batch_normalization_32_23992batch_normalization_32_23994*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_2394320
.batch_normalization_32/StatefulPartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_41_24018conv2d_41_24020*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_240072#
!conv2d_41/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_240282
activation_41/PartitionedCall?
 max_pooling2d_33/PartitionedCallPartitionedCall&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_235382"
 max_pooling2d_33/PartitionedCall?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_33_24101batch_normalization_33_24103batch_normalization_33_24105batch_normalization_33_24107*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2405620
.batch_normalization_33/StatefulPartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_42_24131conv2d_42_24133*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_241202#
!conv2d_42/StatefulPartitionedCall?
activation_42/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_42_layer_call_and_return_conditional_losses_241412
activation_42/PartitionedCall?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_43_24170conv2d_43_24172*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_241592#
!conv2d_43/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_241802
activation_43/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_236542"
 max_pooling2d_34/PartitionedCall?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_34_24253batch_normalization_34_24255batch_normalization_34_24257batch_normalization_34_24259*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2420820
.batch_normalization_34/StatefulPartitionedCall?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0conv2d_44_24283conv2d_44_24285*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_242722#
!conv2d_44/StatefulPartitionedCall?
activation_44/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_44_layer_call_and_return_conditional_losses_242932
activation_44/PartitionedCall?
 max_pooling2d_35/PartitionedCallPartitionedCall&activation_44/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_237702"
 max_pooling2d_35/PartitionedCall?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_35_24366batch_normalization_35_24368batch_normalization_35_24370batch_normalization_35_24372*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2432120
.batch_normalization_35/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_243812
flatten_8/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_16_24411dense_16_24413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_244002"
 dense_16/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_244282#
!dropout_8/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_17_24468dense_17_24470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_244572"
 dense_17/StatefulPartitionedCall?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_40_input
?
d
H__inference_activation_40_layer_call_and_return_conditional_losses_25339

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????\\2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????\\2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????\\:W S
/
_output_shapes
:?????????\\
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_23838

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_40_layer_call_fn_25334

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_238942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????\\2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25864

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_8_layer_call_fn_25983

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_243812
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?.
!__inference__traced_restore_26581
file_prefix%
!assignvariableop_conv2d_40_kernel%
!assignvariableop_1_conv2d_40_bias3
/assignvariableop_2_batch_normalization_32_gamma2
.assignvariableop_3_batch_normalization_32_beta9
5assignvariableop_4_batch_normalization_32_moving_mean=
9assignvariableop_5_batch_normalization_32_moving_variance'
#assignvariableop_6_conv2d_41_kernel%
!assignvariableop_7_conv2d_41_bias3
/assignvariableop_8_batch_normalization_33_gamma2
.assignvariableop_9_batch_normalization_33_beta:
6assignvariableop_10_batch_normalization_33_moving_mean>
:assignvariableop_11_batch_normalization_33_moving_variance(
$assignvariableop_12_conv2d_42_kernel&
"assignvariableop_13_conv2d_42_bias(
$assignvariableop_14_conv2d_43_kernel&
"assignvariableop_15_conv2d_43_bias4
0assignvariableop_16_batch_normalization_34_gamma3
/assignvariableop_17_batch_normalization_34_beta:
6assignvariableop_18_batch_normalization_34_moving_mean>
:assignvariableop_19_batch_normalization_34_moving_variance(
$assignvariableop_20_conv2d_44_kernel&
"assignvariableop_21_conv2d_44_bias4
0assignvariableop_22_batch_normalization_35_gamma3
/assignvariableop_23_batch_normalization_35_beta:
6assignvariableop_24_batch_normalization_35_moving_mean>
:assignvariableop_25_batch_normalization_35_moving_variance'
#assignvariableop_26_dense_16_kernel%
!assignvariableop_27_dense_16_bias'
#assignvariableop_28_dense_17_kernel%
!assignvariableop_29_dense_17_bias!
assignvariableop_30_adam_iter#
assignvariableop_31_adam_beta_1#
assignvariableop_32_adam_beta_2"
assignvariableop_33_adam_decay*
&assignvariableop_34_adam_learning_rate
assignvariableop_35_total
assignvariableop_36_count
assignvariableop_37_total_1
assignvariableop_38_count_1/
+assignvariableop_39_adam_conv2d_40_kernel_m-
)assignvariableop_40_adam_conv2d_40_bias_m;
7assignvariableop_41_adam_batch_normalization_32_gamma_m:
6assignvariableop_42_adam_batch_normalization_32_beta_m/
+assignvariableop_43_adam_conv2d_41_kernel_m-
)assignvariableop_44_adam_conv2d_41_bias_m;
7assignvariableop_45_adam_batch_normalization_33_gamma_m:
6assignvariableop_46_adam_batch_normalization_33_beta_m/
+assignvariableop_47_adam_conv2d_42_kernel_m-
)assignvariableop_48_adam_conv2d_42_bias_m/
+assignvariableop_49_adam_conv2d_43_kernel_m-
)assignvariableop_50_adam_conv2d_43_bias_m;
7assignvariableop_51_adam_batch_normalization_34_gamma_m:
6assignvariableop_52_adam_batch_normalization_34_beta_m/
+assignvariableop_53_adam_conv2d_44_kernel_m-
)assignvariableop_54_adam_conv2d_44_bias_m;
7assignvariableop_55_adam_batch_normalization_35_gamma_m:
6assignvariableop_56_adam_batch_normalization_35_beta_m.
*assignvariableop_57_adam_dense_16_kernel_m,
(assignvariableop_58_adam_dense_16_bias_m.
*assignvariableop_59_adam_dense_17_kernel_m,
(assignvariableop_60_adam_dense_17_bias_m/
+assignvariableop_61_adam_conv2d_40_kernel_v-
)assignvariableop_62_adam_conv2d_40_bias_v;
7assignvariableop_63_adam_batch_normalization_32_gamma_v:
6assignvariableop_64_adam_batch_normalization_32_beta_v/
+assignvariableop_65_adam_conv2d_41_kernel_v-
)assignvariableop_66_adam_conv2d_41_bias_v;
7assignvariableop_67_adam_batch_normalization_33_gamma_v:
6assignvariableop_68_adam_batch_normalization_33_beta_v/
+assignvariableop_69_adam_conv2d_42_kernel_v-
)assignvariableop_70_adam_conv2d_42_bias_v/
+assignvariableop_71_adam_conv2d_43_kernel_v-
)assignvariableop_72_adam_conv2d_43_bias_v;
7assignvariableop_73_adam_batch_normalization_34_gamma_v:
6assignvariableop_74_adam_batch_normalization_34_beta_v/
+assignvariableop_75_adam_conv2d_44_kernel_v-
)assignvariableop_76_adam_conv2d_44_bias_v;
7assignvariableop_77_adam_batch_normalization_35_gamma_v:
6assignvariableop_78_adam_batch_normalization_35_beta_v.
*assignvariableop_79_adam_dense_16_kernel_v,
(assignvariableop_80_adam_dense_16_bias_v.
*assignvariableop_81_adam_dense_17_kernel_v,
(assignvariableop_82_adam_dense_17_bias_v
identity_84??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_9?.
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*?-
value?-B?-TB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*b
dtypesX
V2T	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_40_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_40_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_32_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_32_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_32_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_32_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_41_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_41_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_33_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_33_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_33_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_33_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_42_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_42_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_43_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_43_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_34_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_34_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_34_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_34_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_44_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_44_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_batch_normalization_35_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_35_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp6assignvariableop_24_batch_normalization_35_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_batch_normalization_35_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp#assignvariableop_26_dense_16_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp!assignvariableop_27_dense_16_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_17_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp!assignvariableop_29_dense_17_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_beta_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_decayIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_learning_rateIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_40_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_40_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_32_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_32_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_41_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_41_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_33_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_33_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_42_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_42_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_43_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_43_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp7assignvariableop_51_adam_batch_normalization_34_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_34_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_44_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_44_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_batch_normalization_35_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_35_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_16_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_16_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_17_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_17_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_40_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_40_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_32_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_32_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_41_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_41_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_33_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_33_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_42_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_42_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_43_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_43_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp7assignvariableop_73_adam_batch_normalization_34_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_batch_normalization_34_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv2d_44_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv2d_44_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adam_batch_normalization_35_gamma_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_batch_normalization_35_beta_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_16_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_16_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_17_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_17_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_829
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_83Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_83?
Identity_84IdentityIdentity_83:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_84"#
identity_84Identity_84:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
6__inference_batch_normalization_35_layer_call_fn_25959

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_243212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_44_layer_call_and_return_conditional_losses_25825

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_44_layer_call_and_return_conditional_losses_25839

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_16_layer_call_and_return_conditional_losses_24400

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?	
?
C__inference_dense_17_layer_call_and_return_conditional_losses_24457

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_23637

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_35_layer_call_fn_23776

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_237702
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_43_layer_call_and_return_conditional_losses_24180

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25428

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????..:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????..2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????..::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????..
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_26020

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????
2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????
2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_34_layer_call_fn_23660

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_236542
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_23654

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_35_layer_call_fn_25972

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_243392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25707

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_34_layer_call_fn_25802

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_237222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_40_layer_call_and_return_conditional_losses_23894

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????\\2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_25069

inputs,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource2
.batch_normalization_32_readvariableop_resource4
0batch_normalization_32_readvariableop_1_resourceC
?batch_normalization_32_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource2
.batch_normalization_33_readvariableop_resource4
0batch_normalization_33_readvariableop_1_resourceC
?batch_normalization_33_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_42_conv2d_readvariableop_resource-
)conv2d_42_biasadd_readvariableop_resource,
(conv2d_43_conv2d_readvariableop_resource-
)conv2d_43_biasadd_readvariableop_resource2
.batch_normalization_34_readvariableop_resource4
0batch_normalization_34_readvariableop_1_resourceC
?batch_normalization_34_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_44_conv2d_readvariableop_resource-
)conv2d_44_biasadd_readvariableop_resource2
.batch_normalization_35_readvariableop_resource4
0batch_normalization_35_readvariableop_1_resourceC
?batch_normalization_35_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity??%batch_normalization_32/AssignNewValue?'batch_normalization_32/AssignNewValue_1?6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_32/ReadVariableOp?'batch_normalization_32/ReadVariableOp_1?%batch_normalization_33/AssignNewValue?'batch_normalization_33/AssignNewValue_1?6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_33/ReadVariableOp?'batch_normalization_33/ReadVariableOp_1?%batch_normalization_34/AssignNewValue?'batch_normalization_34/AssignNewValue_1?6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_34/ReadVariableOp?'batch_normalization_34/ReadVariableOp_1?%batch_normalization_35/AssignNewValue?'batch_normalization_35/AssignNewValue_1?6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_35/ReadVariableOp?'batch_normalization_35/ReadVariableOp_1? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_40/Conv2D/ReadVariableOp?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\*
paddingVALID*
strides
2
conv2d_40/Conv2D?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\2
conv2d_40/BiasAdd?
activation_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\\2
activation_40/Relu?
max_pooling2d_32/MaxPoolMaxPool activation_40/Relu:activations:0*/
_output_shapes
:?????????..*
ksize
*
paddingVALID*
strides
2
max_pooling2d_32/MaxPool?
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_32/ReadVariableOp?
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_32/ReadVariableOp_1?
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_32/MaxPool:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????..:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_32/FusedBatchNormV3?
%batch_normalization_32/AssignNewValueAssignVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource4batch_normalization_32/FusedBatchNormV3:batch_mean:07^batch_normalization_32/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_32/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_32/AssignNewValue?
'batch_normalization_32/AssignNewValue_1AssignVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_32/FusedBatchNormV3:batch_variance:09^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_32/AssignNewValue_1?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2DConv2D+batch_normalization_32/FusedBatchNormV3:y:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?*
paddingVALID*
strides
2
conv2d_41/Conv2D?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?2
conv2d_41/BiasAdd?
activation_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:?????????**?2
activation_41/Relu?
max_pooling2d_33/MaxPoolMaxPool activation_41/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_33/MaxPool?
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_33/ReadVariableOp?
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_33/ReadVariableOp_1?
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_33/MaxPool:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_33/FusedBatchNormV3?
%batch_normalization_33/AssignNewValueAssignVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource4batch_normalization_33/FusedBatchNormV3:batch_mean:07^batch_normalization_33/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_33/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_33/AssignNewValue?
'batch_normalization_33/AssignNewValue_1AssignVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_33/FusedBatchNormV3:batch_variance:09^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_33/AssignNewValue_1?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2DConv2D+batch_normalization_33/FusedBatchNormV3:y:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_42/Conv2D?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_42/BiasAdd?
activation_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_42/Relu?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_43/Conv2D/ReadVariableOp?
conv2d_43/Conv2DConv2D activation_42/Relu:activations:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_43/Conv2D?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_43/BiasAdd/ReadVariableOp?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_43/BiasAdd?
activation_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_43/Relu?
max_pooling2d_34/MaxPoolMaxPool activation_43/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_34/MaxPool?
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_34/ReadVariableOp?
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_34/ReadVariableOp_1?
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_34/MaxPool:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_34/FusedBatchNormV3?
%batch_normalization_34/AssignNewValueAssignVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource4batch_normalization_34/FusedBatchNormV3:batch_mean:07^batch_normalization_34/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_34/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_34/AssignNewValue?
'batch_normalization_34/AssignNewValue_1AssignVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_34/FusedBatchNormV3:batch_variance:09^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_34/AssignNewValue_1?
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_44/Conv2D/ReadVariableOp?
conv2d_44/Conv2DConv2D+batch_normalization_34/FusedBatchNormV3:y:0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_44/Conv2D?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_44/BiasAdd/ReadVariableOp?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_44/BiasAdd?
activation_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
activation_44/Relu?
max_pooling2d_35/MaxPoolMaxPool activation_44/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_35/MaxPool?
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_35/ReadVariableOp?
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_35/ReadVariableOp_1?
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_35/MaxPool:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_35/FusedBatchNormV3?
%batch_normalization_35/AssignNewValueAssignVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource4batch_normalization_35/FusedBatchNormV3:batch_mean:07^batch_normalization_35/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_35/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_35/AssignNewValue?
'batch_normalization_35/AssignNewValue_1AssignVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_35/FusedBatchNormV3:batch_variance:09^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_35/AssignNewValue_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_8/Const?
flatten_8/ReshapeReshape+batch_normalization_35/FusedBatchNormV3:y:0flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten_8/Reshape?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	?	
*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulflatten_8/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_16/Reluw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMuldense_16/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*'
_output_shapes
:?????????
2
dropout_8/dropout/Mul}
dropout_8/dropout/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
2
dropout_8/dropout/Mul_1?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldropout_8/dropout/Mul_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Sigmoid?
IdentityIdentitydense_17/Sigmoid:y:0&^batch_normalization_32/AssignNewValue(^batch_normalization_32/AssignNewValue_17^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1&^batch_normalization_33/AssignNewValue(^batch_normalization_33/AssignNewValue_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1&^batch_normalization_34/AssignNewValue(^batch_normalization_34/AssignNewValue_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1&^batch_normalization_35/AssignNewValue(^batch_normalization_35/AssignNewValue_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2N
%batch_normalization_32/AssignNewValue%batch_normalization_32/AssignNewValue2R
'batch_normalization_32/AssignNewValue_1'batch_normalization_32/AssignNewValue_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12N
%batch_normalization_33/AssignNewValue%batch_normalization_33/AssignNewValue2R
'batch_normalization_33/AssignNewValue_1'batch_normalization_33/AssignNewValue_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12N
%batch_normalization_34/AssignNewValue%batch_normalization_34/AssignNewValue2R
'batch_normalization_34/AssignNewValue_1'batch_normalization_34/AssignNewValue_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12N
%batch_normalization_35/AssignNewValue%batch_normalization_35/AssignNewValue2R
'batch_normalization_35/AssignNewValue_1'batch_normalization_35/AssignNewValue_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?g
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_24649

inputs
conv2d_40_24566
conv2d_40_24568 
batch_normalization_32_24573 
batch_normalization_32_24575 
batch_normalization_32_24577 
batch_normalization_32_24579
conv2d_41_24582
conv2d_41_24584 
batch_normalization_33_24589 
batch_normalization_33_24591 
batch_normalization_33_24593 
batch_normalization_33_24595
conv2d_42_24598
conv2d_42_24600
conv2d_43_24604
conv2d_43_24606 
batch_normalization_34_24611 
batch_normalization_34_24613 
batch_normalization_34_24615 
batch_normalization_34_24617
conv2d_44_24620
conv2d_44_24622 
batch_normalization_35_24627 
batch_normalization_35_24629 
batch_normalization_35_24631 
batch_normalization_35_24633
dense_16_24637
dense_16_24639
dense_17_24643
dense_17_24645
identity??.batch_normalization_32/StatefulPartitionedCall?.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_44/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_24566conv2d_40_24568*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_238942#
!conv2d_40/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_239152
activation_40/PartitionedCall?
 max_pooling2d_32/PartitionedCallPartitionedCall&activation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_234222"
 max_pooling2d_32/PartitionedCall?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0batch_normalization_32_24573batch_normalization_32_24575batch_normalization_32_24577batch_normalization_32_24579*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_2394320
.batch_normalization_32/StatefulPartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_41_24582conv2d_41_24584*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_240072#
!conv2d_41/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_240282
activation_41/PartitionedCall?
 max_pooling2d_33/PartitionedCallPartitionedCall&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_235382"
 max_pooling2d_33/PartitionedCall?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_33_24589batch_normalization_33_24591batch_normalization_33_24593batch_normalization_33_24595*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2405620
.batch_normalization_33/StatefulPartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_42_24598conv2d_42_24600*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_241202#
!conv2d_42/StatefulPartitionedCall?
activation_42/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_42_layer_call_and_return_conditional_losses_241412
activation_42/PartitionedCall?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_43_24604conv2d_43_24606*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_241592#
!conv2d_43/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_241802
activation_43/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_236542"
 max_pooling2d_34/PartitionedCall?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_34_24611batch_normalization_34_24613batch_normalization_34_24615batch_normalization_34_24617*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2420820
.batch_normalization_34/StatefulPartitionedCall?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0conv2d_44_24620conv2d_44_24622*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_242722#
!conv2d_44/StatefulPartitionedCall?
activation_44/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_44_layer_call_and_return_conditional_losses_242932
activation_44/PartitionedCall?
 max_pooling2d_35/PartitionedCallPartitionedCall&activation_44/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_237702"
 max_pooling2d_35/PartitionedCall?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_35_24627batch_normalization_35_24629batch_normalization_35_24631batch_normalization_35_24633*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2432120
.batch_normalization_35/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_243812
flatten_8/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_16_24637dense_16_24639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_244002"
 dense_16/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_244282#
!dropout_8/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_17_24643dense_17_24645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_244572"
 dense_17/StatefulPartitionedCall?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_24208

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_8_layer_call_fn_24712
conv2d_40_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_246492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_40_input
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25946

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_42_layer_call_and_return_conditional_losses_25653

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_44_layer_call_and_return_conditional_losses_24272

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_8_layer_call_and_return_conditional_losses_25978

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_23770

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
I
-__inference_activation_43_layer_call_fn_25687

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_241802
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_43_layer_call_and_return_conditional_losses_25682

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_32_layer_call_fn_25395

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_234902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25928

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_43_layer_call_fn_25677

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_241592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?f
?

G__inference_sequential_8_layer_call_and_return_conditional_losses_24560
conv2d_40_input
conv2d_40_24477
conv2d_40_24479 
batch_normalization_32_24484 
batch_normalization_32_24486 
batch_normalization_32_24488 
batch_normalization_32_24490
conv2d_41_24493
conv2d_41_24495 
batch_normalization_33_24500 
batch_normalization_33_24502 
batch_normalization_33_24504 
batch_normalization_33_24506
conv2d_42_24509
conv2d_42_24511
conv2d_43_24515
conv2d_43_24517 
batch_normalization_34_24522 
batch_normalization_34_24524 
batch_normalization_34_24526 
batch_normalization_34_24528
conv2d_44_24531
conv2d_44_24533 
batch_normalization_35_24538 
batch_normalization_35_24540 
batch_normalization_35_24542 
batch_normalization_35_24544
dense_16_24548
dense_16_24550
dense_17_24554
dense_17_24556
identity??.batch_normalization_32/StatefulPartitionedCall?.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_44/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputconv2d_40_24477conv2d_40_24479*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_238942#
!conv2d_40/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_239152
activation_40/PartitionedCall?
 max_pooling2d_32/PartitionedCallPartitionedCall&activation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_234222"
 max_pooling2d_32/PartitionedCall?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0batch_normalization_32_24484batch_normalization_32_24486batch_normalization_32_24488batch_normalization_32_24490*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????..*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_2396120
.batch_normalization_32/StatefulPartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_41_24493conv2d_41_24495*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_240072#
!conv2d_41/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_240282
activation_41/PartitionedCall?
 max_pooling2d_33/PartitionedCallPartitionedCall&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_235382"
 max_pooling2d_33/PartitionedCall?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_33_24500batch_normalization_33_24502batch_normalization_33_24504batch_normalization_33_24506*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2407420
.batch_normalization_33/StatefulPartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_42_24509conv2d_42_24511*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_241202#
!conv2d_42/StatefulPartitionedCall?
activation_42/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_42_layer_call_and_return_conditional_losses_241412
activation_42/PartitionedCall?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_43_24515conv2d_43_24517*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_241592#
!conv2d_43/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_241802
activation_43/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_236542"
 max_pooling2d_34/PartitionedCall?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_34_24522batch_normalization_34_24524batch_normalization_34_24526batch_normalization_34_24528*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2422620
.batch_normalization_34/StatefulPartitionedCall?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0conv2d_44_24531conv2d_44_24533*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_242722#
!conv2d_44/StatefulPartitionedCall?
activation_44/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_44_layer_call_and_return_conditional_losses_242932
activation_44/PartitionedCall?
 max_pooling2d_35/PartitionedCallPartitionedCall&activation_44/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_237702"
 max_pooling2d_35/PartitionedCall?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_35_24538batch_normalization_35_24540batch_normalization_35_24542batch_normalization_35_24544*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2433920
.batch_normalization_35/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_243812
flatten_8/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_16_24548dense_16_24550*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_244002"
 dense_16/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_244332
dropout_8/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_17_24554dense_17_24556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_244572"
 dense_17/StatefulPartitionedCall?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_40_input
?
I
-__inference_activation_41_layer_call_fn_25501

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????**?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_240282
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????**?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????**?:X T
0
_output_shapes
:?????????**?
 
_user_specified_nameinputs
?
I
-__inference_activation_40_layer_call_fn_25344

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????\\* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_239152
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????\\2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????\\:W S
/
_output_shapes
:?????????\\
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_24056

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_34_layer_call_fn_25751

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_242262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_23416
conv2d_40_input9
5sequential_8_conv2d_40_conv2d_readvariableop_resource:
6sequential_8_conv2d_40_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_32_readvariableop_resourceA
=sequential_8_batch_normalization_32_readvariableop_1_resourceP
Lsequential_8_batch_normalization_32_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_41_conv2d_readvariableop_resource:
6sequential_8_conv2d_41_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_33_readvariableop_resourceA
=sequential_8_batch_normalization_33_readvariableop_1_resourceP
Lsequential_8_batch_normalization_33_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_42_conv2d_readvariableop_resource:
6sequential_8_conv2d_42_biasadd_readvariableop_resource9
5sequential_8_conv2d_43_conv2d_readvariableop_resource:
6sequential_8_conv2d_43_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_34_readvariableop_resourceA
=sequential_8_batch_normalization_34_readvariableop_1_resourceP
Lsequential_8_batch_normalization_34_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_44_conv2d_readvariableop_resource:
6sequential_8_conv2d_44_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_35_readvariableop_resourceA
=sequential_8_batch_normalization_35_readvariableop_1_resourceP
Lsequential_8_batch_normalization_35_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource8
4sequential_8_dense_16_matmul_readvariableop_resource9
5sequential_8_dense_16_biasadd_readvariableop_resource8
4sequential_8_dense_17_matmul_readvariableop_resource9
5sequential_8_dense_17_biasadd_readvariableop_resource
identity??Csequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_32/ReadVariableOp?4sequential_8/batch_normalization_32/ReadVariableOp_1?Csequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_33/ReadVariableOp?4sequential_8/batch_normalization_33/ReadVariableOp_1?Csequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_34/ReadVariableOp?4sequential_8/batch_normalization_34/ReadVariableOp_1?Csequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_35/ReadVariableOp?4sequential_8/batch_normalization_35/ReadVariableOp_1?-sequential_8/conv2d_40/BiasAdd/ReadVariableOp?,sequential_8/conv2d_40/Conv2D/ReadVariableOp?-sequential_8/conv2d_41/BiasAdd/ReadVariableOp?,sequential_8/conv2d_41/Conv2D/ReadVariableOp?-sequential_8/conv2d_42/BiasAdd/ReadVariableOp?,sequential_8/conv2d_42/Conv2D/ReadVariableOp?-sequential_8/conv2d_43/BiasAdd/ReadVariableOp?,sequential_8/conv2d_43/Conv2D/ReadVariableOp?-sequential_8/conv2d_44/BiasAdd/ReadVariableOp?,sequential_8/conv2d_44/Conv2D/ReadVariableOp?,sequential_8/dense_16/BiasAdd/ReadVariableOp?+sequential_8/dense_16/MatMul/ReadVariableOp?,sequential_8/dense_17/BiasAdd/ReadVariableOp?+sequential_8/dense_17/MatMul/ReadVariableOp?
,sequential_8/conv2d_40/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,sequential_8/conv2d_40/Conv2D/ReadVariableOp?
sequential_8/conv2d_40/Conv2DConv2Dconv2d_40_input4sequential_8/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\*
paddingVALID*
strides
2
sequential_8/conv2d_40/Conv2D?
-sequential_8/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_8/conv2d_40/BiasAdd/ReadVariableOp?
sequential_8/conv2d_40/BiasAddBiasAdd&sequential_8/conv2d_40/Conv2D:output:05sequential_8/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????\\2 
sequential_8/conv2d_40/BiasAdd?
sequential_8/activation_40/ReluRelu'sequential_8/conv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????\\2!
sequential_8/activation_40/Relu?
%sequential_8/max_pooling2d_32/MaxPoolMaxPool-sequential_8/activation_40/Relu:activations:0*/
_output_shapes
:?????????..*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_32/MaxPool?
2sequential_8/batch_normalization_32/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_8/batch_normalization_32/ReadVariableOp?
4sequential_8/batch_normalization_32/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_8/batch_normalization_32/ReadVariableOp_1?
Csequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_32/FusedBatchNormV3FusedBatchNormV3.sequential_8/max_pooling2d_32/MaxPool:output:0:sequential_8/batch_normalization_32/ReadVariableOp:value:0<sequential_8/batch_normalization_32/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????..:::::*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_32/FusedBatchNormV3?
,sequential_8/conv2d_41/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_41_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02.
,sequential_8/conv2d_41/Conv2D/ReadVariableOp?
sequential_8/conv2d_41/Conv2DConv2D8sequential_8/batch_normalization_32/FusedBatchNormV3:y:04sequential_8/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?*
paddingVALID*
strides
2
sequential_8/conv2d_41/Conv2D?
-sequential_8/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_41/BiasAdd/ReadVariableOp?
sequential_8/conv2d_41/BiasAddBiasAdd&sequential_8/conv2d_41/Conv2D:output:05sequential_8/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????**?2 
sequential_8/conv2d_41/BiasAdd?
sequential_8/activation_41/ReluRelu'sequential_8/conv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:?????????**?2!
sequential_8/activation_41/Relu?
%sequential_8/max_pooling2d_33/MaxPoolMaxPool-sequential_8/activation_41/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_33/MaxPool?
2sequential_8/batch_normalization_33/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_33_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_8/batch_normalization_33/ReadVariableOp?
4sequential_8/batch_normalization_33/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_8/batch_normalization_33/ReadVariableOp_1?
Csequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3.sequential_8/max_pooling2d_33/MaxPool:output:0:sequential_8/batch_normalization_33/ReadVariableOp:value:0<sequential_8/batch_normalization_33/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_33/FusedBatchNormV3?
,sequential_8/conv2d_42/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,sequential_8/conv2d_42/Conv2D/ReadVariableOp?
sequential_8/conv2d_42/Conv2DConv2D8sequential_8/batch_normalization_33/FusedBatchNormV3:y:04sequential_8/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential_8/conv2d_42/Conv2D?
-sequential_8/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_42/BiasAdd/ReadVariableOp?
sequential_8/conv2d_42/BiasAddBiasAdd&sequential_8/conv2d_42/Conv2D:output:05sequential_8/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_8/conv2d_42/BiasAdd?
sequential_8/activation_42/ReluRelu'sequential_8/conv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2!
sequential_8/activation_42/Relu?
,sequential_8/conv2d_43/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,sequential_8/conv2d_43/Conv2D/ReadVariableOp?
sequential_8/conv2d_43/Conv2DConv2D-sequential_8/activation_42/Relu:activations:04sequential_8/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential_8/conv2d_43/Conv2D?
-sequential_8/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_43/BiasAdd/ReadVariableOp?
sequential_8/conv2d_43/BiasAddBiasAdd&sequential_8/conv2d_43/Conv2D:output:05sequential_8/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_8/conv2d_43/BiasAdd?
sequential_8/activation_43/ReluRelu'sequential_8/conv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2!
sequential_8/activation_43/Relu?
%sequential_8/max_pooling2d_34/MaxPoolMaxPool-sequential_8/activation_43/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_34/MaxPool?
2sequential_8/batch_normalization_34/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_34_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_8/batch_normalization_34/ReadVariableOp?
4sequential_8/batch_normalization_34/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_8/batch_normalization_34/ReadVariableOp_1?
Csequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_34/FusedBatchNormV3FusedBatchNormV3.sequential_8/max_pooling2d_34/MaxPool:output:0:sequential_8/batch_normalization_34/ReadVariableOp:value:0<sequential_8/batch_normalization_34/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_34/FusedBatchNormV3?
,sequential_8/conv2d_44/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,sequential_8/conv2d_44/Conv2D/ReadVariableOp?
sequential_8/conv2d_44/Conv2DConv2D8sequential_8/batch_normalization_34/FusedBatchNormV3:y:04sequential_8/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential_8/conv2d_44/Conv2D?
-sequential_8/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_44/BiasAdd/ReadVariableOp?
sequential_8/conv2d_44/BiasAddBiasAdd&sequential_8/conv2d_44/Conv2D:output:05sequential_8/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_8/conv2d_44/BiasAdd?
sequential_8/activation_44/ReluRelu'sequential_8/conv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2!
sequential_8/activation_44/Relu?
%sequential_8/max_pooling2d_35/MaxPoolMaxPool-sequential_8/activation_44/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_35/MaxPool?
2sequential_8/batch_normalization_35/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_35_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_8/batch_normalization_35/ReadVariableOp?
4sequential_8/batch_normalization_35/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_8/batch_normalization_35/ReadVariableOp_1?
Csequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_35/FusedBatchNormV3FusedBatchNormV3.sequential_8/max_pooling2d_35/MaxPool:output:0:sequential_8/batch_normalization_35/ReadVariableOp:value:0<sequential_8/batch_normalization_35/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_35/FusedBatchNormV3?
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential_8/flatten_8/Const?
sequential_8/flatten_8/ReshapeReshape8sequential_8/batch_normalization_35/FusedBatchNormV3:y:0%sequential_8/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????	2 
sequential_8/flatten_8/Reshape?
+sequential_8/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_16_matmul_readvariableop_resource*
_output_shapes
:	?	
*
dtype02-
+sequential_8/dense_16/MatMul/ReadVariableOp?
sequential_8/dense_16/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_8/dense_16/MatMul?
,sequential_8/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_16_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_8/dense_16/BiasAdd/ReadVariableOp?
sequential_8/dense_16/BiasAddBiasAdd&sequential_8/dense_16/MatMul:product:04sequential_8/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_8/dense_16/BiasAdd?
sequential_8/dense_16/ReluRelu&sequential_8/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential_8/dense_16/Relu?
sequential_8/dropout_8/IdentityIdentity(sequential_8/dense_16/Relu:activations:0*
T0*'
_output_shapes
:?????????
2!
sequential_8/dropout_8/Identity?
+sequential_8/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_17_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02-
+sequential_8/dense_17/MatMul/ReadVariableOp?
sequential_8/dense_17/MatMulMatMul(sequential_8/dropout_8/Identity:output:03sequential_8/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_17/MatMul?
,sequential_8/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_8/dense_17/BiasAdd/ReadVariableOp?
sequential_8/dense_17/BiasAddBiasAdd&sequential_8/dense_17/MatMul:product:04sequential_8/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_17/BiasAdd?
sequential_8/dense_17/SigmoidSigmoid&sequential_8/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_17/Sigmoid?
IdentityIdentity!sequential_8/dense_17/Sigmoid:y:0D^sequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_32/ReadVariableOp5^sequential_8/batch_normalization_32/ReadVariableOp_1D^sequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_33/ReadVariableOp5^sequential_8/batch_normalization_33/ReadVariableOp_1D^sequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_34/ReadVariableOp5^sequential_8/batch_normalization_34/ReadVariableOp_1D^sequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_35/ReadVariableOp5^sequential_8/batch_normalization_35/ReadVariableOp_1.^sequential_8/conv2d_40/BiasAdd/ReadVariableOp-^sequential_8/conv2d_40/Conv2D/ReadVariableOp.^sequential_8/conv2d_41/BiasAdd/ReadVariableOp-^sequential_8/conv2d_41/Conv2D/ReadVariableOp.^sequential_8/conv2d_42/BiasAdd/ReadVariableOp-^sequential_8/conv2d_42/Conv2D/ReadVariableOp.^sequential_8/conv2d_43/BiasAdd/ReadVariableOp-^sequential_8/conv2d_43/Conv2D/ReadVariableOp.^sequential_8/conv2d_44/BiasAdd/ReadVariableOp-^sequential_8/conv2d_44/Conv2D/ReadVariableOp-^sequential_8/dense_16/BiasAdd/ReadVariableOp,^sequential_8/dense_16/MatMul/ReadVariableOp-^sequential_8/dense_17/BiasAdd/ReadVariableOp,^sequential_8/dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2?
Csequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_32/ReadVariableOp2sequential_8/batch_normalization_32/ReadVariableOp2l
4sequential_8/batch_normalization_32/ReadVariableOp_14sequential_8/batch_normalization_32/ReadVariableOp_12?
Csequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_33/ReadVariableOp2sequential_8/batch_normalization_33/ReadVariableOp2l
4sequential_8/batch_normalization_33/ReadVariableOp_14sequential_8/batch_normalization_33/ReadVariableOp_12?
Csequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_34/ReadVariableOp2sequential_8/batch_normalization_34/ReadVariableOp2l
4sequential_8/batch_normalization_34/ReadVariableOp_14sequential_8/batch_normalization_34/ReadVariableOp_12?
Csequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_35/ReadVariableOp2sequential_8/batch_normalization_35/ReadVariableOp2l
4sequential_8/batch_normalization_35/ReadVariableOp_14sequential_8/batch_normalization_35/ReadVariableOp_12^
-sequential_8/conv2d_40/BiasAdd/ReadVariableOp-sequential_8/conv2d_40/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_40/Conv2D/ReadVariableOp,sequential_8/conv2d_40/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_41/BiasAdd/ReadVariableOp-sequential_8/conv2d_41/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_41/Conv2D/ReadVariableOp,sequential_8/conv2d_41/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_42/BiasAdd/ReadVariableOp-sequential_8/conv2d_42/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_42/Conv2D/ReadVariableOp,sequential_8/conv2d_42/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_43/BiasAdd/ReadVariableOp-sequential_8/conv2d_43/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_43/Conv2D/ReadVariableOp,sequential_8/conv2d_43/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_44/BiasAdd/ReadVariableOp-sequential_8/conv2d_44/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_44/Conv2D/ReadVariableOp,sequential_8/conv2d_44/Conv2D/ReadVariableOp2\
,sequential_8/dense_16/BiasAdd/ReadVariableOp,sequential_8/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_16/MatMul/ReadVariableOp+sequential_8/dense_16/MatMul/ReadVariableOp2\
,sequential_8/dense_17/BiasAdd/ReadVariableOp,sequential_8/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_17/MatMul/ReadVariableOp+sequential_8/dense_17/MatMul/ReadVariableOp:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_40_input
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25521

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_23943

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????..:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????..2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????..::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????..
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_34_layer_call_fn_25815

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_237532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_24226

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_41_layer_call_and_return_conditional_losses_24028

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????**?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????**?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????**?:X T
0
_output_shapes
:?????????**?
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_35_layer_call_fn_25895

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_238382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_24321

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
conv2d_40_inputB
!serving_default_conv2d_40_input:0???????????<
dense_170
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ȯ
??
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer-20
layer_with_weights-10
layer-21
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"??
_tf_keras_sequential??{"class_name": "Sequential", "name": "sequential_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 375, 375, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_40_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 375, 375, 3]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_40", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_32", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_41", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_33", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_42", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_43", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_43", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_34", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_44", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_44", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_35", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 375, 375, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 375, 375, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_40_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 375, 375, 3]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_40", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_32", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_41", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_33", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_42", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_43", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_43", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_34", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_44", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_44", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_35", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "BinaryCrossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 375, 375, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_40", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 375, 375, 3]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 375, 375, 3]}}
?
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_40", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_32", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0trainable_variables
1	variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_32", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_32", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 8]}}
?	

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 8]}}
?
:trainable_variables
;	variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_41", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
>trainable_variables
?	variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_33", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_33", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 128]}}
?	

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 128]}}
?
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_42", "trainable": true, "dtype": "float32", "activation": "relu"}}
?	

Ukernel
Vbias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_43", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 19, 256]}}
?
[trainable_variables
\	variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_43", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
_trainable_variables
`	variables
aregularization_losses
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
htrainable_variables
i	variables
jregularization_losses
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_34", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_34", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}
?	

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_44", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}
?
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_44", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
zaxis
	{gamma
|beta
}moving_mean
~moving_variance
trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_35", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_35", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 128]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?,m?-m?4m?5m?Cm?Dm?Km?Lm?Um?Vm?dm?em?lm?mm?{m?|m?	?m?	?m?	?m?	?m?v?v?,v?-v?4v?5v?Cv?Dv?Kv?Lv?Uv?Vv?dv?ev?lv?mv?{v?|v?	?v?	?v?	?v?	?v?"
	optimizer
?
0
1
,2
-3
44
55
C6
D7
K8
L9
U10
V11
d12
e13
l14
m15
{16
|17
?18
?19
?20
?21"
trackable_list_wrapper
?
0
1
,2
-3
.4
/5
46
57
C8
D9
E10
F11
K12
L13
U14
V15
d16
e17
f18
g19
l20
m21
{22
|23
}24
~25
?26
?27
?28
?29"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
	variables
 ?layer_regularization_losses
regularization_losses
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(2conv2d_40/kernel
:2conv2d_40/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
 	variables
 ?layer_regularization_losses
!regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
$	variables
 ?layer_regularization_losses
%regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
(	variables
 ?layer_regularization_losses
)regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_32/gamma
):'2batch_normalization_32/beta
2:0 (2"batch_normalization_32/moving_mean
6:4 (2&batch_normalization_32/moving_variance
.
,0
-1"
trackable_list_wrapper
<
,0
-1
.2
/3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
1	variables
 ?layer_regularization_losses
2regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?2conv2d_41/kernel
:?2conv2d_41/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
7	variables
 ?layer_regularization_losses
8regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
:trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
;	variables
 ?layer_regularization_losses
<regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
>trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
@regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_33/gamma
*:(?2batch_normalization_33/beta
3:1? (2"batch_normalization_33/moving_mean
7:5? (2&batch_normalization_33/moving_variance
.
C0
D1"
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
H	variables
 ?layer_regularization_losses
Iregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_42/kernel
:?2conv2d_42/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
N	variables
 ?layer_regularization_losses
Oregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
R	variables
 ?layer_regularization_losses
Sregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_43/kernel
:?2conv2d_43/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
X	variables
 ?layer_regularization_losses
Yregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
[trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
\	variables
 ?layer_regularization_losses
]regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
_trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
`	variables
 ?layer_regularization_losses
aregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_34/gamma
*:(?2batch_normalization_34/beta
3:1? (2"batch_normalization_34/moving_mean
7:5? (2&batch_normalization_34/moving_variance
.
d0
e1"
trackable_list_wrapper
<
d0
e1
f2
g3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
htrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
i	variables
 ?layer_regularization_losses
jregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_44/kernel
:?2conv2d_44/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ntrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
o	variables
 ?layer_regularization_losses
pregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
s	variables
 ?layer_regularization_losses
tregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
vtrainable_variables
?metrics
?layer_metrics
?non_trainable_variables
w	variables
 ?layer_regularization_losses
xregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_35/gamma
*:(?2batch_normalization_35/beta
3:1? (2"batch_normalization_35/moving_mean
7:5? (2&batch_normalization_35/moving_variance
.
{0
|1"
trackable_list_wrapper
<
{0
|1
}2
~3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?	
2dense_16/kernel
:
2dense_16/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_17/kernel
:2dense_17/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?layer_metrics
?non_trainable_variables
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
X
.0
/1
E2
F3
f4
g5
}6
~7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-2Adam/conv2d_40/kernel/m
!:2Adam/conv2d_40/bias/m
/:-2#Adam/batch_normalization_32/gamma/m
.:,2"Adam/batch_normalization_32/beta/m
0:.?2Adam/conv2d_41/kernel/m
": ?2Adam/conv2d_41/bias/m
0:.?2#Adam/batch_normalization_33/gamma/m
/:-?2"Adam/batch_normalization_33/beta/m
1:/??2Adam/conv2d_42/kernel/m
": ?2Adam/conv2d_42/bias/m
1:/??2Adam/conv2d_43/kernel/m
": ?2Adam/conv2d_43/bias/m
0:.?2#Adam/batch_normalization_34/gamma/m
/:-?2"Adam/batch_normalization_34/beta/m
1:/??2Adam/conv2d_44/kernel/m
": ?2Adam/conv2d_44/bias/m
0:.?2#Adam/batch_normalization_35/gamma/m
/:-?2"Adam/batch_normalization_35/beta/m
':%	?	
2Adam/dense_16/kernel/m
 :
2Adam/dense_16/bias/m
&:$
2Adam/dense_17/kernel/m
 :2Adam/dense_17/bias/m
/:-2Adam/conv2d_40/kernel/v
!:2Adam/conv2d_40/bias/v
/:-2#Adam/batch_normalization_32/gamma/v
.:,2"Adam/batch_normalization_32/beta/v
0:.?2Adam/conv2d_41/kernel/v
": ?2Adam/conv2d_41/bias/v
0:.?2#Adam/batch_normalization_33/gamma/v
/:-?2"Adam/batch_normalization_33/beta/v
1:/??2Adam/conv2d_42/kernel/v
": ?2Adam/conv2d_42/bias/v
1:/??2Adam/conv2d_43/kernel/v
": ?2Adam/conv2d_43/bias/v
0:.?2#Adam/batch_normalization_34/gamma/v
/:-?2"Adam/batch_normalization_34/beta/v
1:/??2Adam/conv2d_44/kernel/v
": ?2Adam/conv2d_44/bias/v
0:.?2#Adam/batch_normalization_35/gamma/v
/:-?2"Adam/batch_normalization_35/beta/v
':%	?	
2Adam/dense_16/kernel/v
 :
2Adam/dense_16/bias/v
&:$
2Adam/dense_17/kernel/v
 :2Adam/dense_17/bias/v
?2?
,__inference_sequential_8_layer_call_fn_25250
,__inference_sequential_8_layer_call_fn_25315
,__inference_sequential_8_layer_call_fn_24863
,__inference_sequential_8_layer_call_fn_24712?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_8_layer_call_and_return_conditional_losses_25069
G__inference_sequential_8_layer_call_and_return_conditional_losses_24560
G__inference_sequential_8_layer_call_and_return_conditional_losses_24474
G__inference_sequential_8_layer_call_and_return_conditional_losses_25185?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_23416?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0
conv2d_40_input???????????
?2?
)__inference_conv2d_40_layer_call_fn_25334?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_40_layer_call_and_return_conditional_losses_25325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_40_layer_call_fn_25344?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_40_layer_call_and_return_conditional_losses_25339?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_32_layer_call_fn_23428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_23422?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
6__inference_batch_normalization_32_layer_call_fn_25459
6__inference_batch_normalization_32_layer_call_fn_25472
6__inference_batch_normalization_32_layer_call_fn_25408
6__inference_batch_normalization_32_layer_call_fn_25395?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25364
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25382
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25428
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25446?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_41_layer_call_fn_25491?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_41_layer_call_and_return_conditional_losses_25482?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_41_layer_call_fn_25501?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_41_layer_call_and_return_conditional_losses_25496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_33_layer_call_fn_23544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_23538?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
6__inference_batch_normalization_33_layer_call_fn_25552
6__inference_batch_normalization_33_layer_call_fn_25565
6__inference_batch_normalization_33_layer_call_fn_25629
6__inference_batch_normalization_33_layer_call_fn_25616?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25603
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25585
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25521
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25539?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_42_layer_call_fn_25648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_42_layer_call_and_return_conditional_losses_25639?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_42_layer_call_fn_25658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_42_layer_call_and_return_conditional_losses_25653?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_43_layer_call_fn_25677?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_25668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_43_layer_call_fn_25687?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_43_layer_call_and_return_conditional_losses_25682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_34_layer_call_fn_23660?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_23654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
6__inference_batch_normalization_34_layer_call_fn_25815
6__inference_batch_normalization_34_layer_call_fn_25738
6__inference_batch_normalization_34_layer_call_fn_25751
6__inference_batch_normalization_34_layer_call_fn_25802?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25707
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25725
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25771
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25789?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_44_layer_call_fn_25834?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_44_layer_call_and_return_conditional_losses_25825?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_44_layer_call_fn_25844?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_44_layer_call_and_return_conditional_losses_25839?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_35_layer_call_fn_23776?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_23770?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
6__inference_batch_normalization_35_layer_call_fn_25895
6__inference_batch_normalization_35_layer_call_fn_25908
6__inference_batch_normalization_35_layer_call_fn_25972
6__inference_batch_normalization_35_layer_call_fn_25959?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25928
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25882
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25864
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25946?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_flatten_8_layer_call_fn_25983?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_8_layer_call_and_return_conditional_losses_25978?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_16_layer_call_fn_26003?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_16_layer_call_and_return_conditional_losses_25994?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dropout_8_layer_call_fn_26030
)__inference_dropout_8_layer_call_fn_26025?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_8_layer_call_and_return_conditional_losses_26020
D__inference_dropout_8_layer_call_and_return_conditional_losses_26015?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_17_layer_call_fn_26050?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_17_layer_call_and_return_conditional_losses_26041?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_24938conv2d_40_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_23416?",-./45CDEFKLUVdefglm{|}~????B??
8?5
3?0
conv2d_40_input???????????
? "3?0
.
dense_17"?
dense_17??????????
H__inference_activation_40_layer_call_and_return_conditional_losses_25339h7?4
-?*
(?%
inputs?????????\\
? "-?*
#? 
0?????????\\
? ?
-__inference_activation_40_layer_call_fn_25344[7?4
-?*
(?%
inputs?????????\\
? " ??????????\\?
H__inference_activation_41_layer_call_and_return_conditional_losses_25496j8?5
.?+
)?&
inputs?????????**?
? ".?+
$?!
0?????????**?
? ?
-__inference_activation_41_layer_call_fn_25501]8?5
.?+
)?&
inputs?????????**?
? "!??????????**??
H__inference_activation_42_layer_call_and_return_conditional_losses_25653j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_activation_42_layer_call_fn_25658]8?5
.?+
)?&
inputs??????????
? "!????????????
H__inference_activation_43_layer_call_and_return_conditional_losses_25682j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_activation_43_layer_call_fn_25687]8?5
.?+
)?&
inputs??????????
? "!????????????
H__inference_activation_44_layer_call_and_return_conditional_losses_25839j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_activation_44_layer_call_fn_25844]8?5
.?+
)?&
inputs??????????
? "!????????????
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25364?,-./M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25382?,-./M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25428r,-./;?8
1?.
(?%
inputs?????????..
p
? "-?*
#? 
0?????????..
? ?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_25446r,-./;?8
1?.
(?%
inputs?????????..
p 
? "-?*
#? 
0?????????..
? ?
6__inference_batch_normalization_32_layer_call_fn_25395?,-./M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_32_layer_call_fn_25408?,-./M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_32_layer_call_fn_25459e,-./;?8
1?.
(?%
inputs?????????..
p
? " ??????????..?
6__inference_batch_normalization_32_layer_call_fn_25472e,-./;?8
1?.
(?%
inputs?????????..
p 
? " ??????????..?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25521tCDEF<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25539tCDEF<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25585?CDEFN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_25603?CDEFN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_33_layer_call_fn_25552gCDEF<?9
2?/
)?&
inputs??????????
p
? "!????????????
6__inference_batch_normalization_33_layer_call_fn_25565gCDEF<?9
2?/
)?&
inputs??????????
p 
? "!????????????
6__inference_batch_normalization_33_layer_call_fn_25616?CDEFN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_33_layer_call_fn_25629?CDEFN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25707tdefg<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25725tdefg<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25771?defgN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_25789?defgN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_34_layer_call_fn_25738gdefg<?9
2?/
)?&
inputs??????????
p
? "!????????????
6__inference_batch_normalization_34_layer_call_fn_25751gdefg<?9
2?/
)?&
inputs??????????
p 
? "!????????????
6__inference_batch_normalization_34_layer_call_fn_25802?defgN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_34_layer_call_fn_25815?defgN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25864?{|}~N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25882?{|}~N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25928t{|}~<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_25946t{|}~<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
6__inference_batch_normalization_35_layer_call_fn_25895?{|}~N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_35_layer_call_fn_25908?{|}~N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_35_layer_call_fn_25959g{|}~<?9
2?/
)?&
inputs??????????
p
? "!????????????
6__inference_batch_normalization_35_layer_call_fn_25972g{|}~<?9
2?/
)?&
inputs??????????
p 
? "!????????????
D__inference_conv2d_40_layer_call_and_return_conditional_losses_25325n9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????\\
? ?
)__inference_conv2d_40_layer_call_fn_25334a9?6
/?,
*?'
inputs???????????
? " ??????????\\?
D__inference_conv2d_41_layer_call_and_return_conditional_losses_25482m457?4
-?*
(?%
inputs?????????..
? ".?+
$?!
0?????????**?
? ?
)__inference_conv2d_41_layer_call_fn_25491`457?4
-?*
(?%
inputs?????????..
? "!??????????**??
D__inference_conv2d_42_layer_call_and_return_conditional_losses_25639nKL8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_42_layer_call_fn_25648aKL8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_conv2d_43_layer_call_and_return_conditional_losses_25668nUV8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_43_layer_call_fn_25677aUV8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_conv2d_44_layer_call_and_return_conditional_losses_25825nlm8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_44_layer_call_fn_25834alm8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_dense_16_layer_call_and_return_conditional_losses_25994_??0?-
&?#
!?
inputs??????????	
? "%?"
?
0?????????

? ~
(__inference_dense_16_layer_call_fn_26003R??0?-
&?#
!?
inputs??????????	
? "??????????
?
C__inference_dense_17_layer_call_and_return_conditional_losses_26041^??/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? }
(__inference_dense_17_layer_call_fn_26050Q??/?,
%?"
 ?
inputs?????????

? "???????????
D__inference_dropout_8_layer_call_and_return_conditional_losses_26015\3?0
)?&
 ?
inputs?????????

p
? "%?"
?
0?????????

? ?
D__inference_dropout_8_layer_call_and_return_conditional_losses_26020\3?0
)?&
 ?
inputs?????????

p 
? "%?"
?
0?????????

? |
)__inference_dropout_8_layer_call_fn_26025O3?0
)?&
 ?
inputs?????????

p
? "??????????
|
)__inference_dropout_8_layer_call_fn_26030O3?0
)?&
 ?
inputs?????????

p 
? "??????????
?
D__inference_flatten_8_layer_call_and_return_conditional_losses_25978b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????	
? ?
)__inference_flatten_8_layer_call_fn_25983U8?5
.?+
)?&
inputs??????????
? "???????????	?
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_23422?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_32_layer_call_fn_23428?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_23538?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_33_layer_call_fn_23544?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_23654?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_34_layer_call_fn_23660?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_23770?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_35_layer_call_fn_23776?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_sequential_8_layer_call_and_return_conditional_losses_24474?",-./45CDEFKLUVdefglm{|}~????J?G
@?=
3?0
conv2d_40_input???????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_24560?",-./45CDEFKLUVdefglm{|}~????J?G
@?=
3?0
conv2d_40_input???????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_25069?",-./45CDEFKLUVdefglm{|}~????A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_25185?",-./45CDEFKLUVdefglm{|}~????A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_8_layer_call_fn_24712?",-./45CDEFKLUVdefglm{|}~????J?G
@?=
3?0
conv2d_40_input???????????
p

 
? "???????????
,__inference_sequential_8_layer_call_fn_24863?",-./45CDEFKLUVdefglm{|}~????J?G
@?=
3?0
conv2d_40_input???????????
p 

 
? "???????????
,__inference_sequential_8_layer_call_fn_25250?",-./45CDEFKLUVdefglm{|}~????A?>
7?4
*?'
inputs???????????
p

 
? "???????????
,__inference_sequential_8_layer_call_fn_25315?",-./45CDEFKLUVdefglm{|}~????A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
#__inference_signature_wrapper_24938?",-./45CDEFKLUVdefglm{|}~????U?R
? 
K?H
F
conv2d_40_input3?0
conv2d_40_input???????????"3?0
.
dense_17"?
dense_17?????????