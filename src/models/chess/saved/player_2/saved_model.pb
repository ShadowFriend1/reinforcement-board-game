йК
Ш┼
.
Abs
x"T
y"T"
Ttype:

2	
Ъ
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(љ
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
│
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.11.02v2.11.0-rc2-15-g6290819256d8иг
Ѕ
Player2QNet/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ѓ *)
shared_namePlayer2QNet/dense_8/bias
ѓ
,Player2QNet/dense_8/bias/Read/ReadVariableOpReadVariableOpPlayer2QNet/dense_8/bias*
_output_shapes	
:ѓ *
dtype0
Љ
Player2QNet/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(ѓ *+
shared_namePlayer2QNet/dense_8/kernel
і
.Player2QNet/dense_8/kernel/Read/ReadVariableOpReadVariableOpPlayer2QNet/dense_8/kernel*
_output_shapes
:	(ѓ *
dtype0
е
(Player2QNet/EncodingNetwork/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*9
shared_name*(Player2QNet/EncodingNetwork/dense_7/bias
А
<Player2QNet/EncodingNetwork/dense_7/bias/Read/ReadVariableOpReadVariableOp(Player2QNet/EncodingNetwork/dense_7/bias*
_output_shapes
:(*
dtype0
░
*Player2QNet/EncodingNetwork/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:K(*;
shared_name,*Player2QNet/EncodingNetwork/dense_7/kernel
Е
>Player2QNet/EncodingNetwork/dense_7/kernel/Read/ReadVariableOpReadVariableOp*Player2QNet/EncodingNetwork/dense_7/kernel*
_output_shapes

:K(*
dtype0
е
(Player2QNet/EncodingNetwork/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:K*9
shared_name*(Player2QNet/EncodingNetwork/dense_6/bias
А
<Player2QNet/EncodingNetwork/dense_6/bias/Read/ReadVariableOpReadVariableOp(Player2QNet/EncodingNetwork/dense_6/bias*
_output_shapes
:K*
dtype0
░
*Player2QNet/EncodingNetwork/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@K*;
shared_name,*Player2QNet/EncodingNetwork/dense_6/kernel
Е
>Player2QNet/EncodingNetwork/dense_6/kernel/Read/ReadVariableOpReadVariableOp*Player2QNet/EncodingNetwork/dense_6/kernel*
_output_shapes

:@K*
dtype0
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
l
action_0_discountPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
~
action_0_observation_maskPlaceholder*(
_output_shapes
:         ѓ *
dtype0*
shape:         ѓ 
Ё
action_0_observation_statePlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
j
action_0_rewardPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
m
action_0_step_typePlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
щ
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observation_maskaction_0_observation_stateaction_0_rewardaction_0_step_type*Player2QNet/EncodingNetwork/dense_6/kernel(Player2QNet/EncodingNetwork/dense_6/bias*Player2QNet/EncodingNetwork/dense_7/kernel(Player2QNet/EncodingNetwork/dense_7/biasPlayer2QNet/dense_8/kernelPlayer2QNet/dense_8/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *(
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *0
f+R)
'__inference_signature_wrapper_123552312
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ч
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *0
f+R)
'__inference_signature_wrapper_123552317
П
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *0
f+R)
'__inference_signature_wrapper_123552329
ў
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *0
f+R)
'__inference_signature_wrapper_123552325

NoOpNoOp
в)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*д)
valueю)BЎ) Bњ)
╩
collect_data_spec

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
	get_metadata

get_train_step

signatures*

observation
1* 
GA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*
D
_time_step_spec
_trajectory_spec
_wrapped_policy*

trace_0
trace_1* 

trace_0* 

trace_0* 
* 
* 
K

action
get_initial_state
get_train_step
get_metadata* 
* 
pj
VARIABLE_VALUE*Player2QNet/EncodingNetwork/dense_6/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE(Player2QNet/EncodingNetwork/dense_6/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE*Player2QNet/EncodingNetwork/dense_7/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE(Player2QNet/EncodingNetwork/dense_7/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEPlayer2QNet/dense_8/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEPlayer2QNet/dense_8/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*

observation
3* 

observation
1* 
?

_q_network
_time_step_spec
 _trajectory_spec*
* 
* 
* 
* 
* 
* 
* 
* 
┤
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_input_tensor_spec

(_q_net*

)observation
)3* 

)observation
)1* 
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
Њ
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
* 
▓
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_encoder
6_q_value_layer*
* 
* 

(0*
* 
* 
* 
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
Њ
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
г
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_postprocessing_layers*
д
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias*
* 

50
61*
* 
* 
* 
 
0
1
2
3*
 
0
1
2
3*
* 
Њ
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
* 
* 

N0
O1
P2*

0
1*

0
1*
* 
Њ
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 
* 

N0
O1
P2*
* 
* 
* 
ј
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
д
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

kernel
bias*
д
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

kernel
bias*
* 
* 
* 
* 
* 
* 
* 
* 
Љ
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 
Њ
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
Њ
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ю
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp>Player2QNet/EncodingNetwork/dense_6/kernel/Read/ReadVariableOp<Player2QNet/EncodingNetwork/dense_6/bias/Read/ReadVariableOp>Player2QNet/EncodingNetwork/dense_7/kernel/Read/ReadVariableOp<Player2QNet/EncodingNetwork/dense_7/bias/Read/ReadVariableOp.Player2QNet/dense_8/kernel/Read/ReadVariableOp,Player2QNet/dense_8/bias/Read/ReadVariableOpConst*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__traced_save_123552379
ї
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable*Player2QNet/EncodingNetwork/dense_6/kernel(Player2QNet/EncodingNetwork/dense_6/bias*Player2QNet/EncodingNetwork/dense_7/kernel(Player2QNet/EncodingNetwork/dense_7/biasPlayer2QNet/dense_8/kernelPlayer2QNet/dense_8/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *.
f)R'
%__inference__traced_restore_123552410иО
К
>
,__inference_function_with_signature_70676808

batch_size 
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ */
f*R(
&__inference_get_initial_state_70676807*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Э
╠
'__inference_signature_wrapper_123552312
discount
observation_mask
observation_state

reward
	step_type
unknown:@K
	unknown_0:K
	unknown_1:K(
	unknown_2:(
	unknown_3:	(ѓ 
	unknown_4:	ѓ 
identityѕбStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_maskobservation_stateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *(
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *5
f0R.
,__inference_function_with_signature_70676778k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         ѓ :         :         :         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:         
$
_user_specified_name
0/discount:\X
(
_output_shapes
:         ѓ 
,
_user_specified_name0/observation/mask:`\
+
_output_shapes
:         
-
_user_specified_name0/observation/state:MI
#
_output_shapes
:         
"
_user_specified_name
0/reward:PL
#
_output_shapes
:         
%
_user_specified_name0/step_type
м
.
,__inference_function_with_signature_70676831Р
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *!
fR
__inference_<lambda>_714*(
_construction_contextkEagerRuntime*
_input_shapes 
╚
9
'__inference_signature_wrapper_123552317

batch_sizeЁ
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *5
f0R.
,__inference_function_with_signature_70676808*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ЭA
ы
0__inference_polymorphic_distribution_fn_70677020
	step_type

reward
discount
observation_mask
observation_state`
Nplayer2qnet_player2qnet_encodingnetwork_dense_6_matmul_readvariableop_resource:@K]
Oplayer2qnet_player2qnet_encodingnetwork_dense_6_biasadd_readvariableop_resource:K`
Nplayer2qnet_player2qnet_encodingnetwork_dense_7_matmul_readvariableop_resource:K(]
Oplayer2qnet_player2qnet_encodingnetwork_dense_7_biasadd_readvariableop_resource:(Q
>player2qnet_player2qnet_dense_8_matmul_readvariableop_resource:	(ѓ N
?player2qnet_player2qnet_dense_8_biasadd_readvariableop_resource:	ѓ 
identity

identity_1

identity_2ѕбFPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpбEPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpбFPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpбEPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpб6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOpб5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOpѕ
7Player2QNet/Player2QNet/EncodingNetwork/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   ╦
9Player2QNet/Player2QNet/EncodingNetwork/flatten_2/ReshapeReshapeobservation_state@Player2QNet/Player2QNet/EncodingNetwork/flatten_2/Const:output:0*
T0*'
_output_shapes
:         @┴
4Player2QNet/Player2QNet/EncodingNetwork/dense_6/CastCastBPlayer2QNet/Player2QNet/EncodingNetwork/flatten_2/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:         @н
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpReadVariableOpNplayer2qnet_player2qnet_encodingnetwork_dense_6_matmul_readvariableop_resource*
_output_shapes

:@K*
dtype0ч
6Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMulMatMul8Player2QNet/Player2QNet/EncodingNetwork/dense_6/Cast:y:0MPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Kм
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpReadVariableOpOplayer2qnet_player2qnet_encodingnetwork_dense_6_biasadd_readvariableop_resource*
_output_shapes
:K*
dtype0є
7Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAddBiasAdd@Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul:product:0NPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         K░
4Player2QNet/Player2QNet/EncodingNetwork/dense_6/ReluRelu@Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         Kн
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpNplayer2qnet_player2qnet_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:K(*
dtype0Ё
6Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMulMatMulBPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/Relu:activations:0MPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (м
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpOplayer2qnet_player2qnet_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0є
7Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAddBiasAdd@Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul:product:0NPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (░
4Player2QNet/Player2QNet/EncodingNetwork/dense_7/ReluRelu@Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         (х
5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOpReadVariableOp>player2qnet_player2qnet_dense_8_matmul_readvariableop_resource*
_output_shapes
:	(ѓ *
dtype0Т
&Player2QNet/Player2QNet/dense_8/MatMulMatMulBPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/Relu:activations:0=Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ѓ │
6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOpReadVariableOp?player2qnet_player2qnet_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ѓ *
dtype0О
'Player2QNet/Player2QNet/dense_8/BiasAddBiasAdd0Player2QNet/Player2QNet/dense_8/MatMul:product:0>Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ѓ q
Player2QNet/ShapeShape0Player2QNet/Player2QNet/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:Y
Player2QNet/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : і
Player2QNet/zerosFillPlayer2QNet/Shape:output:0 Player2QNet/zeros/Const:output:0*
T0*(
_output_shapes
:         ѓ {
Player2QNet/EqualEqualPlayer2QNet/zeros:output:0observation_mask*
T0*(
_output_shapes
:         ѓ {
Player2QNet/AbsAbs0Player2QNet/Player2QNet/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ѓ [
Player2QNet/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *    а
Player2QNet/SelectV2SelectV2Player2QNet/Equal:z:0Player2QNet/SelectV2/t:output:0Player2QNet/Abs:y:0*
T0*(
_output_shapes
:         ѓ l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         џ
Categorical/mode/ArgMaxArgMaxPlayer2QNet/SelectV2:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         |
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: f

Identity_1IdentityCategorical/mode/Cast:y:0^NoOp*
T0*#
_output_shapes
:         [

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: ┘
NoOpNoOpG^Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpF^Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpG^Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpF^Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp7^Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp6^Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         :         :         ѓ :         : : : : : : 2љ
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpFPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp2ј
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpEPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOp2љ
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpFPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2ј
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpEPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp2p
6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp2n
5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp:N J
#
_output_shapes
:         
#
_user_specified_name	step_type:KG
#
_output_shapes
:         
 
_user_specified_namereward:MI
#
_output_shapes
:         
"
_user_specified_name
discount:ZV
(
_output_shapes
:         ѓ 
*
_user_specified_nameobservation/mask:^Z
+
_output_shapes
:         
+
_user_specified_nameobservation/state
▄
g
'__inference_signature_wrapper_123552325
unknown:	 
identity	ѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *5
f0R.
,__inference_function_with_signature_70676820^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
═
l
,__inference_function_with_signature_70676820
unknown:	 
identity	ѕбStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *!
fR
__inference_<lambda>_711^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
М^
§
*__inference_polymorphic_action_fn_70676975
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_mask
time_step_observation_state`
Nplayer2qnet_player2qnet_encodingnetwork_dense_6_matmul_readvariableop_resource:@K]
Oplayer2qnet_player2qnet_encodingnetwork_dense_6_biasadd_readvariableop_resource:K`
Nplayer2qnet_player2qnet_encodingnetwork_dense_7_matmul_readvariableop_resource:K(]
Oplayer2qnet_player2qnet_encodingnetwork_dense_7_biasadd_readvariableop_resource:(Q
>player2qnet_player2qnet_dense_8_matmul_readvariableop_resource:	(ѓ N
?player2qnet_player2qnet_dense_8_biasadd_readvariableop_resource:	ѓ 
identityѕбFPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpбEPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpбFPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpбEPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpб6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOpб5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOpѕ
7Player2QNet/Player2QNet/EncodingNetwork/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   Н
9Player2QNet/Player2QNet/EncodingNetwork/flatten_2/ReshapeReshapetime_step_observation_state@Player2QNet/Player2QNet/EncodingNetwork/flatten_2/Const:output:0*
T0*'
_output_shapes
:         @┴
4Player2QNet/Player2QNet/EncodingNetwork/dense_6/CastCastBPlayer2QNet/Player2QNet/EncodingNetwork/flatten_2/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:         @н
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpReadVariableOpNplayer2qnet_player2qnet_encodingnetwork_dense_6_matmul_readvariableop_resource*
_output_shapes

:@K*
dtype0ч
6Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMulMatMul8Player2QNet/Player2QNet/EncodingNetwork/dense_6/Cast:y:0MPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Kм
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpReadVariableOpOplayer2qnet_player2qnet_encodingnetwork_dense_6_biasadd_readvariableop_resource*
_output_shapes
:K*
dtype0є
7Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAddBiasAdd@Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul:product:0NPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         K░
4Player2QNet/Player2QNet/EncodingNetwork/dense_6/ReluRelu@Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         Kн
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpNplayer2qnet_player2qnet_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:K(*
dtype0Ё
6Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMulMatMulBPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/Relu:activations:0MPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (м
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpOplayer2qnet_player2qnet_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0є
7Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAddBiasAdd@Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul:product:0NPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (░
4Player2QNet/Player2QNet/EncodingNetwork/dense_7/ReluRelu@Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         (х
5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOpReadVariableOp>player2qnet_player2qnet_dense_8_matmul_readvariableop_resource*
_output_shapes
:	(ѓ *
dtype0Т
&Player2QNet/Player2QNet/dense_8/MatMulMatMulBPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/Relu:activations:0=Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ѓ │
6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOpReadVariableOp?player2qnet_player2qnet_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ѓ *
dtype0О
'Player2QNet/Player2QNet/dense_8/BiasAddBiasAdd0Player2QNet/Player2QNet/dense_8/MatMul:product:0>Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ѓ q
Player2QNet/ShapeShape0Player2QNet/Player2QNet/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:Y
Player2QNet/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : і
Player2QNet/zerosFillPlayer2QNet/Shape:output:0 Player2QNet/zeros/Const:output:0*
T0*(
_output_shapes
:         ѓ Ё
Player2QNet/EqualEqualPlayer2QNet/zeros:output:0time_step_observation_mask*
T0*(
_output_shapes
:         ѓ {
Player2QNet/AbsAbs0Player2QNet/Player2QNet/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ѓ [
Player2QNet/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *    а
Player2QNet/SelectV2SelectV2Player2QNet/Equal:z:0Player2QNet/SelectV2/t:output:0Player2QNet/Abs:y:0*
T0*(
_output_shapes
:         ѓ l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         џ
Categorical/mode/ArgMaxArgMaxPlayer2QNet/SelectV2:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         |
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB c
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB «
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:б
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:         u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:└
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:е
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:         Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B :Ђ Ќ
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         Q
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         \
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:         ┘
NoOpNoOpG^Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpF^Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpG^Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpF^Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp7^Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp6^Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         :         :         ѓ :         : : : : : : 2љ
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpFPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp2ј
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpEPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOp2љ
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpFPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2ј
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpEPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp2p
6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp2n
5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp:X T
#
_output_shapes
:         
-
_user_specified_nametime_step_step_type:UQ
#
_output_shapes
:         
*
_user_specified_nametime_step_reward:WS
#
_output_shapes
:         
,
_user_specified_nametime_step_discount:d`
(
_output_shapes
:         ѓ 
4
_user_specified_nametime_step_observation_mask:hd
+
_output_shapes
:         
5
_user_specified_nametime_step_observation_state
Ы
_
__inference_<lambda>_711!
readvariableop_resource:	 
identity	ѕбReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
ч
Л
,__inference_function_with_signature_70676778
	step_type

reward
discount
observation_mask
observation_state
unknown:@K
	unknown_0:K
	unknown_1:K(
	unknown_2:(
	unknown_3:	(ѓ 
	unknown_4:	ѓ 
identityѕбStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_maskobservation_stateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *(
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *3
f.R,
*__inference_polymorphic_action_fn_70676763k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         :         :         ѓ :         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:         
%
_user_specified_name0/step_type:MI
#
_output_shapes
:         
"
_user_specified_name
0/reward:OK
#
_output_shapes
:         
$
_user_specified_name
0/discount:\X
(
_output_shapes
:         ѓ 
,
_user_specified_name0/observation/mask:`\
+
_output_shapes
:         
-
_user_specified_name0/observation/state
д$
и
%__inference__traced_restore_123552410
file_prefix#
assignvariableop_variable:	 O
=assignvariableop_1_player2qnet_encodingnetwork_dense_6_kernel:@KI
;assignvariableop_2_player2qnet_encodingnetwork_dense_6_bias:KO
=assignvariableop_3_player2qnet_encodingnetwork_dense_7_kernel:K(I
;assignvariableop_4_player2qnet_encodingnetwork_dense_7_bias:(@
-assignvariableop_5_player2qnet_dense_8_kernel:	(ѓ :
+assignvariableop_6_player2qnet_dense_8_bias:	ѓ 

identity_8ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6╚
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ь
valueСBрB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHђ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B к
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:г
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_1AssignVariableOp=assignvariableop_1_player2qnet_encodingnetwork_dense_6_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_2AssignVariableOp;assignvariableop_2_player2qnet_encodingnetwork_dense_6_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_3AssignVariableOp=assignvariableop_3_player2qnet_encodingnetwork_dense_7_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_4AssignVariableOp;assignvariableop_4_player2qnet_encodingnetwork_dense_7_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_5AssignVariableOp-assignvariableop_5_player2qnet_dense_8_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_6AssignVariableOp+assignvariableop_6_player2qnet_dense_8_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 в

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: ┘
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*"
_acd_function_control_output(*
_output_shapes
 "!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
┴]
╚
*__inference_polymorphic_action_fn_70676763
	time_step
time_step_1
time_step_2
time_step_3
time_step_4`
Nplayer2qnet_player2qnet_encodingnetwork_dense_6_matmul_readvariableop_resource:@K]
Oplayer2qnet_player2qnet_encodingnetwork_dense_6_biasadd_readvariableop_resource:K`
Nplayer2qnet_player2qnet_encodingnetwork_dense_7_matmul_readvariableop_resource:K(]
Oplayer2qnet_player2qnet_encodingnetwork_dense_7_biasadd_readvariableop_resource:(Q
>player2qnet_player2qnet_dense_8_matmul_readvariableop_resource:	(ѓ N
?player2qnet_player2qnet_dense_8_biasadd_readvariableop_resource:	ѓ 
identityѕбFPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpбEPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpбFPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpбEPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpб6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOpб5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOpѕ
7Player2QNet/Player2QNet/EncodingNetwork/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   ┼
9Player2QNet/Player2QNet/EncodingNetwork/flatten_2/ReshapeReshapetime_step_4@Player2QNet/Player2QNet/EncodingNetwork/flatten_2/Const:output:0*
T0*'
_output_shapes
:         @┴
4Player2QNet/Player2QNet/EncodingNetwork/dense_6/CastCastBPlayer2QNet/Player2QNet/EncodingNetwork/flatten_2/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:         @н
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpReadVariableOpNplayer2qnet_player2qnet_encodingnetwork_dense_6_matmul_readvariableop_resource*
_output_shapes

:@K*
dtype0ч
6Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMulMatMul8Player2QNet/Player2QNet/EncodingNetwork/dense_6/Cast:y:0MPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Kм
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpReadVariableOpOplayer2qnet_player2qnet_encodingnetwork_dense_6_biasadd_readvariableop_resource*
_output_shapes
:K*
dtype0є
7Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAddBiasAdd@Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul:product:0NPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         K░
4Player2QNet/Player2QNet/EncodingNetwork/dense_6/ReluRelu@Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         Kн
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpNplayer2qnet_player2qnet_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:K(*
dtype0Ё
6Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMulMatMulBPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/Relu:activations:0MPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (м
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpOplayer2qnet_player2qnet_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0є
7Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAddBiasAdd@Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul:product:0NPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (░
4Player2QNet/Player2QNet/EncodingNetwork/dense_7/ReluRelu@Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         (х
5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOpReadVariableOp>player2qnet_player2qnet_dense_8_matmul_readvariableop_resource*
_output_shapes
:	(ѓ *
dtype0Т
&Player2QNet/Player2QNet/dense_8/MatMulMatMulBPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/Relu:activations:0=Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ѓ │
6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOpReadVariableOp?player2qnet_player2qnet_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ѓ *
dtype0О
'Player2QNet/Player2QNet/dense_8/BiasAddBiasAdd0Player2QNet/Player2QNet/dense_8/MatMul:product:0>Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ѓ q
Player2QNet/ShapeShape0Player2QNet/Player2QNet/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:Y
Player2QNet/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : і
Player2QNet/zerosFillPlayer2QNet/Shape:output:0 Player2QNet/zeros/Const:output:0*
T0*(
_output_shapes
:         ѓ v
Player2QNet/EqualEqualPlayer2QNet/zeros:output:0time_step_3*
T0*(
_output_shapes
:         ѓ {
Player2QNet/AbsAbs0Player2QNet/Player2QNet/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ѓ [
Player2QNet/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *    а
Player2QNet/SelectV2SelectV2Player2QNet/Equal:z:0Player2QNet/SelectV2/t:output:0Player2QNet/Abs:y:0*
T0*(
_output_shapes
:         ѓ l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         џ
Categorical/mode/ArgMaxArgMaxPlayer2QNet/SelectV2:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         |
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB c
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB «
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:б
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:         u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:└
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:е
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:         Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B :Ђ Ќ
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         Q
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         \
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:         ┘
NoOpNoOpG^Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpF^Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpG^Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpF^Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp7^Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp6^Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         :         :         ѓ :         : : : : : : 2љ
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpFPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp2ј
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpEPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOp2љ
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpFPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2ј
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpEPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp2p
6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp2n
5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp:N J
#
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:SO
(
_output_shapes
:         ѓ 
#
_user_specified_name	time_step:VR
+
_output_shapes
:         
#
_user_specified_name	time_step
¤
┤
"__inference__traced_save_123552379
file_prefix'
#savev2_variable_read_readvariableop	I
Esavev2_player2qnet_encodingnetwork_dense_6_kernel_read_readvariableopG
Csavev2_player2qnet_encodingnetwork_dense_6_bias_read_readvariableopI
Esavev2_player2qnet_encodingnetwork_dense_7_kernel_read_readvariableopG
Csavev2_player2qnet_encodingnetwork_dense_7_bias_read_readvariableop9
5savev2_player2qnet_dense_8_kernel_read_readvariableop7
3savev2_player2qnet_dense_8_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ┼
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ь
valueСBрB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B ѕ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopEsavev2_player2qnet_encodingnetwork_dense_6_kernel_read_readvariableopCsavev2_player2qnet_encodingnetwork_dense_6_bias_read_readvariableopEsavev2_player2qnet_encodingnetwork_dense_7_kernel_read_readvariableopCsavev2_player2qnet_encodingnetwork_dense_7_bias_read_readvariableop5savev2_player2qnet_dense_8_kernel_read_readvariableop3savev2_player2qnet_dense_8_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes

2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*K
_input_shapes:
8: : :@K:K:K(:(:	(ѓ :ѓ : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:@K: 

_output_shapes
:K:$ 

_output_shapes

:K(: 

_output_shapes
:(:%!

_output_shapes
:	(ѓ :!

_output_shapes	
:ѓ :

_output_shapes
: 
┐
8
&__inference_get_initial_state_70677023

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
р
)
'__inference_signature_wrapper_123552329Ш
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *5
f0R.
,__inference_function_with_signature_70676831*(
_construction_contextkEagerRuntime*
_input_shapes 
┌]
╦
*__inference_polymorphic_action_fn_70676905
	step_type

reward
discount
observation_mask
observation_state`
Nplayer2qnet_player2qnet_encodingnetwork_dense_6_matmul_readvariableop_resource:@K]
Oplayer2qnet_player2qnet_encodingnetwork_dense_6_biasadd_readvariableop_resource:K`
Nplayer2qnet_player2qnet_encodingnetwork_dense_7_matmul_readvariableop_resource:K(]
Oplayer2qnet_player2qnet_encodingnetwork_dense_7_biasadd_readvariableop_resource:(Q
>player2qnet_player2qnet_dense_8_matmul_readvariableop_resource:	(ѓ N
?player2qnet_player2qnet_dense_8_biasadd_readvariableop_resource:	ѓ 
identityѕбFPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpбEPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpбFPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpбEPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpб6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOpб5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOpѕ
7Player2QNet/Player2QNet/EncodingNetwork/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   ╦
9Player2QNet/Player2QNet/EncodingNetwork/flatten_2/ReshapeReshapeobservation_state@Player2QNet/Player2QNet/EncodingNetwork/flatten_2/Const:output:0*
T0*'
_output_shapes
:         @┴
4Player2QNet/Player2QNet/EncodingNetwork/dense_6/CastCastBPlayer2QNet/Player2QNet/EncodingNetwork/flatten_2/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:         @н
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpReadVariableOpNplayer2qnet_player2qnet_encodingnetwork_dense_6_matmul_readvariableop_resource*
_output_shapes

:@K*
dtype0ч
6Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMulMatMul8Player2QNet/Player2QNet/EncodingNetwork/dense_6/Cast:y:0MPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Kм
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpReadVariableOpOplayer2qnet_player2qnet_encodingnetwork_dense_6_biasadd_readvariableop_resource*
_output_shapes
:K*
dtype0є
7Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAddBiasAdd@Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul:product:0NPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         K░
4Player2QNet/Player2QNet/EncodingNetwork/dense_6/ReluRelu@Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         Kн
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpNplayer2qnet_player2qnet_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:K(*
dtype0Ё
6Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMulMatMulBPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/Relu:activations:0MPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (м
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpOplayer2qnet_player2qnet_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0є
7Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAddBiasAdd@Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul:product:0NPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (░
4Player2QNet/Player2QNet/EncodingNetwork/dense_7/ReluRelu@Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         (х
5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOpReadVariableOp>player2qnet_player2qnet_dense_8_matmul_readvariableop_resource*
_output_shapes
:	(ѓ *
dtype0Т
&Player2QNet/Player2QNet/dense_8/MatMulMatMulBPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/Relu:activations:0=Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ѓ │
6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOpReadVariableOp?player2qnet_player2qnet_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ѓ *
dtype0О
'Player2QNet/Player2QNet/dense_8/BiasAddBiasAdd0Player2QNet/Player2QNet/dense_8/MatMul:product:0>Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ѓ q
Player2QNet/ShapeShape0Player2QNet/Player2QNet/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:Y
Player2QNet/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : і
Player2QNet/zerosFillPlayer2QNet/Shape:output:0 Player2QNet/zeros/Const:output:0*
T0*(
_output_shapes
:         ѓ {
Player2QNet/EqualEqualPlayer2QNet/zeros:output:0observation_mask*
T0*(
_output_shapes
:         ѓ {
Player2QNet/AbsAbs0Player2QNet/Player2QNet/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ѓ [
Player2QNet/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *    а
Player2QNet/SelectV2SelectV2Player2QNet/Equal:z:0Player2QNet/SelectV2/t:output:0Player2QNet/Abs:y:0*
T0*(
_output_shapes
:         ѓ l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         џ
Categorical/mode/ArgMaxArgMaxPlayer2QNet/SelectV2:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         |
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB c
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB «
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:б
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:         u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:└
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:е
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:         Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B :Ђ Ќ
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         Q
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         \
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:         ┘
NoOpNoOpG^Player2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpF^Player2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpG^Player2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpF^Player2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp7^Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp6^Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         :         :         ѓ :         : : : : : : 2љ
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpFPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp2ј
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOpEPlayer2QNet/Player2QNet/EncodingNetwork/dense_6/MatMul/ReadVariableOp2љ
FPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpFPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2ј
EPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOpEPlayer2QNet/Player2QNet/EncodingNetwork/dense_7/MatMul/ReadVariableOp2p
6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp6Player2QNet/Player2QNet/dense_8/BiasAdd/ReadVariableOp2n
5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp5Player2QNet/Player2QNet/dense_8/MatMul/ReadVariableOp:N J
#
_output_shapes
:         
#
_user_specified_name	step_type:KG
#
_output_shapes
:         
 
_user_specified_namereward:MI
#
_output_shapes
:         
"
_user_specified_name
discount:ZV
(
_output_shapes
:         ѓ 
*
_user_specified_nameobservation/mask:^Z
+
_output_shapes
:         
+
_user_specified_nameobservation/state
Y

__inference_<lambda>_714*(
_construction_contextkEagerRuntime*
_input_shapes 
┐
8
&__inference_get_initial_state_70676807

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size"є
L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*џ
actionЈ
4

0/discount&
action_0_discount:0         
I
0/observation/mask3
action_0_observation_mask:0         ѓ 
N
0/observation/state7
action_0_observation_state:0         
0
0/reward$
action_0_reward:0         
6
0/step_type'
action_0_step_type:0         6
action,
StatefulPartitionedCall:0         tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:╔z
С
collect_data_spec

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
	get_metadata

get_train_step

signatures"
_generic_user_object
9
observation
1"
trackable_tuple_wrapper
:	 (2Variable
 "
trackable_dict_wrapper
K
0
1
2
3
4
5"
trackable_tuple_wrapper
`
_time_step_spec
_trajectory_spec
_wrapped_policy"
trackable_dict_wrapper
├
trace_0
trace_12ї
*__inference_polymorphic_action_fn_70676905
*__inference_polymorphic_action_fn_70676975▒
ф▓д
FullArgSpec(
args џ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsб
б 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0ztrace_1
Ѓ
trace_02Т
0__inference_polymorphic_distribution_fn_70677020▒
ф▓д
FullArgSpec(
args џ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsб
б 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0
Ь
trace_02Л
&__inference_get_initial_state_70677023д
Ю▓Ў
FullArgSpec!
argsџ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0
«BФ
__inference_<lambda>_714"ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
«BФ
__inference_<lambda>_711"ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
`

action
get_initial_state
get_train_step
get_metadata"
signature_map
 "
trackable_dict_wrapper
<::@K2*Player2QNet/EncodingNetwork/dense_6/kernel
6:4K2(Player2QNet/EncodingNetwork/dense_6/bias
<::K(2*Player2QNet/EncodingNetwork/dense_7/kernel
6:4(2(Player2QNet/EncodingNetwork/dense_7/bias
-:+	(ѓ 2Player2QNet/dense_8/kernel
':%ѓ 2Player2QNet/dense_8/bias
9
observation
3"
trackable_tuple_wrapper
9
observation
1"
trackable_tuple_wrapper
Y

_q_network
_time_step_spec
 _trajectory_spec"
_generic_user_object
ДBц
*__inference_polymorphic_action_fn_70676905	step_typerewarddiscountobservation/maskobservation/state"▒
ф▓д
FullArgSpec(
args џ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsб
б 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┘Bо
*__inference_polymorphic_action_fn_70676975time_step_step_typetime_step_rewardtime_step_discounttime_step_observation_masktime_step_observation_state"▒
ф▓д
FullArgSpec(
args џ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsб
б 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ГBф
0__inference_polymorphic_distribution_fn_70677020	step_typerewarddiscountobservation/maskobservation/state"▒
ф▓д
FullArgSpec(
args џ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsб
б 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
РB▀
&__inference_get_initial_state_70677023
batch_size"д
Ю▓Ў
FullArgSpec!
argsџ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
'__inference_signature_wrapper_123552312
0/discount0/observation/mask0/observation/state0/reward0/step_type"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЛB╬
'__inference_signature_wrapper_123552317
batch_size"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
├B└
'__inference_signature_wrapper_123552325"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
├B└
'__inference_signature_wrapper_123552329"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╔
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_input_tensor_spec

(_q_net"
_tf_keras_layer
9
)observation
)3"
trackable_tuple_wrapper
9
)observation
)1"
trackable_tuple_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
ы2Ьв
Р▓я
FullArgSpec@
args8џ5
jself
jobservation
j	step_type
jnetwork_state
varargs
 
varkw
 
defaultsб

 

 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ы2Ьв
Р▓я
FullArgSpec@
args8џ5
jself
jobservation
j	step_type
jnetwork_state
varargs
 
varkw
 
defaultsб

 

 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
 "
trackable_dict_wrapper
К
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_encoder
6_q_value_layer"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
т2Р▀
о▓м
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsб

 
б 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
т2Р▀
о▓м
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsб

 
б 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┴
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_postprocessing_layers"
_tf_keras_layer
╗
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
т2Р▀
о▓м
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsб

 
б 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
т2Р▀
о▓м
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsб

 
б 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
N0
O1
P2"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ц
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╗
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
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
Г
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
trackable_dict_wrapper@
__inference_<lambda>_711$б

б 
ф "і
unknown 	0
__inference_<lambda>_714б

б 
ф "ф S
&__inference_get_initial_state_70677023)"б
б
і

batch_size 
ф "б ║
*__inference_polymorphic_action_fn_70676905Ігбе
абю
ћ▓љ
TimeStep,
	step_typeі
	step_type         &
rewardі
reward         *
discountі
discount         Ђ
observationrфo
3
mask+і(
observation/mask         ѓ 
8
state/і,
observation/state         
б 
ф "R▓O

PolicyStep&
actionі
action         
stateб 
infoб Ь
*__inference_polymorphic_action_fn_70676975┐Яб▄
нбл
╚▓─
TimeStep6
	step_type)і&
time_step_step_type         0
reward&і#
time_step_reward         4
discount(і%
time_step_discount         Ќ
observationЄфЃ
=
mask5і2
time_step_observation_mask         ѓ 
B
state9і6
time_step_observation_state         
б 
ф "R▓O

PolicyStep&
actionі
action         
stateб 
infoб Ъ
0__inference_polymorphic_distribution_fn_70677020Жгбе
абю
ћ▓љ
TimeStep,
	step_typeі
	step_type         &
rewardі
reward         *
discountі
discount         Ђ
observationrфo
3
mask+і(
observation/mask         ѓ 
8
state/і,
observation/state         
б 
ф "░▓г

PolicyStepѓ
actionэњз┐б╗
`
Bф?

atolі 

locі         

rtolі 
LфI

allow_nan_statsp

namejDeterministic_1_1

validate_argsp 
б
j
parameters
б 
б
jname+tfp.distributions.Deterministic_ACTTypeSpec 
stateб 
infoб ┤
'__inference_signature_wrapper_123552312ѕлб╠
б 
─ф└
5

0/discount'і$
tensor_0_discount         
J
0/observation/mask4і1
tensor_0_observation_mask         ѓ 
O
0/observation/state8і5
tensor_0_observation_state         
1
0/reward%і"
tensor_0_reward         
7
0/step_type(і%
tensor_0_step_type         "+ф(
&
actionі
action         b
'__inference_signature_wrapper_12355231770б-
б 
&ф#
!

batch_sizeі

batch_size "ф [
'__inference_signature_wrapper_1235523250б

б 
ф "ф

int64і
int64 	?
'__inference_signature_wrapper_123552329б

б 
ф "ф 