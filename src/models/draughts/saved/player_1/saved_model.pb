й┼
Ў┼
.
Abs
x"T
y"T"
Ttype:

2	
Я
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
validate_shapebool( И
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
incompatible_shape_errorbool(Р
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02v2.11.0-rc2-15-g6290819256d8зк
Й
Player1QNet/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А *)
shared_namePlayer1QNet/dense_2/bias
В
,Player1QNet/dense_2/bias/Read/ReadVariableOpReadVariableOpPlayer1QNet/dense_2/bias*
_output_shapes	
:А *
dtype0
С
Player1QNet/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(А *+
shared_namePlayer1QNet/dense_2/kernel
К
.Player1QNet/dense_2/kernel/Read/ReadVariableOpReadVariableOpPlayer1QNet/dense_2/kernel*
_output_shapes
:	(А *
dtype0
и
(Player1QNet/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*9
shared_name*(Player1QNet/EncodingNetwork/dense_1/bias
б
<Player1QNet/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp(Player1QNet/EncodingNetwork/dense_1/bias*
_output_shapes
:(*
dtype0
░
*Player1QNet/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:K(*;
shared_name,*Player1QNet/EncodingNetwork/dense_1/kernel
й
>Player1QNet/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp*Player1QNet/EncodingNetwork/dense_1/kernel*
_output_shapes

:K(*
dtype0
д
&Player1QNet/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:K*7
shared_name(&Player1QNet/EncodingNetwork/dense/bias
Э
:Player1QNet/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp&Player1QNet/EncodingNetwork/dense/bias*
_output_shapes
:K*
dtype0
м
(Player1QNet/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@K*9
shared_name*(Player1QNet/EncodingNetwork/dense/kernel
е
<Player1QNet/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp(Player1QNet/EncodingNetwork/dense/kernel*
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
:         А *
dtype0*
shape:         А 
Е
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
ї
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observation_maskaction_0_observation_stateaction_0_rewardaction_0_step_type(Player1QNet/EncodingNetwork/dense/kernel&Player1QNet/EncodingNetwork/dense/bias*Player1QNet/EncodingNetwork/dense_1/kernel(Player1QNet/EncodingNetwork/dense_1/biasPlayer1QNet/dense_2/kernelPlayer1QNet/dense_2/bias*
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
GPU 2J 8В *0
f+R)
'__inference_signature_wrapper_259972866
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
№
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
GPU 2J 8В *0
f+R)
'__inference_signature_wrapper_259972871
▌
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
GPU 2J 8В *0
f+R)
'__inference_signature_wrapper_259972883
Ш
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
GPU 2J 8В *0
f+R)
'__inference_signature_wrapper_259972879

NoOpNoOp
ч)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*в)
valueШ)BХ) BО)
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
nh
VARIABLE_VALUE(Player1QNet/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE&Player1QNet/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE*Player1QNet/EncodingNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE(Player1QNet/EncodingNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEPlayer1QNet/dense_2/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEPlayer1QNet/dense_2/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
У
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
У
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
м
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_postprocessing_layers*
ж
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
У
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
У
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
О
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
ж
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

kernel
bias*
ж
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
С
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
У
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
У
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
Щ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp<Player1QNet/EncodingNetwork/dense/kernel/Read/ReadVariableOp:Player1QNet/EncodingNetwork/dense/bias/Read/ReadVariableOp>Player1QNet/EncodingNetwork/dense_1/kernel/Read/ReadVariableOp<Player1QNet/EncodingNetwork/dense_1/bias/Read/ReadVariableOp.Player1QNet/dense_2/kernel/Read/ReadVariableOp,Player1QNet/dense_2/bias/Read/ReadVariableOpConst*
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
GPU 2J 8В *+
f&R$
"__inference__traced_save_259972933
И
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable(Player1QNet/EncodingNetwork/dense/kernel&Player1QNet/EncodingNetwork/dense/bias*Player1QNet/EncodingNetwork/dense_1/kernel(Player1QNet/EncodingNetwork/dense_1/biasPlayer1QNet/dense_2/kernelPlayer1QNet/dense_2/bias*
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
GPU 2J 8В *.
f)R'
%__inference__traced_restore_259972964╟╒
Є
_
__inference_<lambda>_614!
readvariableop_resource:	 
identity	ИвReadVariableOp^
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
╥
.
,__inference_function_with_signature_66915230т
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
GPU 2J 8В *!
fR
__inference_<lambda>_617*(
_construction_contextkEagerRuntime*
_input_shapes 
▄
g
'__inference_signature_wrapper_259972879
unknown:	 
identity	ИвStatefulPartitionedCallЪ
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
GPU 2J 8В *5
f0R.
,__inference_function_with_signature_66915219^
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
┐
8
&__inference_get_initial_state_66915206

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
Ю$
│
%__inference__traced_restore_259972964
file_prefix#
assignvariableop_variable:	 M
;assignvariableop_1_player1qnet_encodingnetwork_dense_kernel:@KG
9assignvariableop_2_player1qnet_encodingnetwork_dense_bias:KO
=assignvariableop_3_player1qnet_encodingnetwork_dense_1_kernel:K(I
;assignvariableop_4_player1qnet_encodingnetwork_dense_1_bias:(@
-assignvariableop_5_player1qnet_dense_2_kernel:	(А :
+assignvariableop_6_player1qnet_dense_2_bias:	А 

identity_8ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6╚
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ю
valueфBсB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHА
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B ╞
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:м
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_1AssignVariableOp;assignvariableop_1_player1qnet_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╨
AssignVariableOp_2AssignVariableOp9assignvariableop_2_player1qnet_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╘
AssignVariableOp_3AssignVariableOp=assignvariableop_3_player1qnet_encodingnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_4AssignVariableOp;assignvariableop_4_player1qnet_encodingnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_5AssignVariableOp-assignvariableop_5_player1qnet_dense_2_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_6AssignVariableOp+assignvariableop_6_player1qnet_dense_2_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ы

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
╚
9
'__inference_signature_wrapper_259972871

batch_sizeЕ
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
GPU 2J 8В *5
f0R.
,__inference_function_with_signature_66915207*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
с
)
'__inference_signature_wrapper_259972883Ў
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
GPU 2J 8В *5
f0R.
,__inference_function_with_signature_66915230*(
_construction_contextkEagerRuntime*
_input_shapes 
°
╠
'__inference_signature_wrapper_259972866
discount
observation_mask
observation_state

reward
	step_type
unknown:@K
	unknown_0:K
	unknown_1:K(
	unknown_2:(
	unknown_3:	(А 
	unknown_4:	А 
identityИвStatefulPartitionedCallп
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
GPU 2J 8В *5
f0R.
,__inference_function_with_signature_66915177k
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
d:         :         А :         :         :         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:         
$
_user_specified_name
0/discount:\X
(
_output_shapes
:         А 
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
═
l
,__inference_function_with_signature_66915219
unknown:	 
identity	ИвStatefulPartitionedCallЖ
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
GPU 2J 8В *!
fR
__inference_<lambda>_614^
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
Y

__inference_<lambda>_617*(
_construction_contextkEagerRuntime*
_input_shapes 
√
╤
,__inference_function_with_signature_66915177
	step_type

reward
discount
observation_mask
observation_state
unknown:@K
	unknown_0:K
	unknown_1:K(
	unknown_2:(
	unknown_3:	(А 
	unknown_4:	А 
identityИвStatefulPartitionedCallн
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
GPU 2J 8В *3
f.R,
*__inference_polymorphic_action_fn_66915162k
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
d:         :         :         :         А :         : : : : : : 22
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
:         А 
,
_user_specified_name0/observation/mask:`\
+
_output_shapes
:         
-
_user_specified_name0/observation/state
╟
>
,__inference_function_with_signature_66915207

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
GPU 2J 8В */
f*R(
&__inference_get_initial_state_66915206*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
┐
8
&__inference_get_initial_state_66915422

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
в]
├
*__inference_polymorphic_action_fn_66915304
	step_type

reward
discount
observation_mask
observation_state^
Lplayer1qnet_player1qnet_encodingnetwork_dense_matmul_readvariableop_resource:@K[
Mplayer1qnet_player1qnet_encodingnetwork_dense_biasadd_readvariableop_resource:K`
Nplayer1qnet_player1qnet_encodingnetwork_dense_1_matmul_readvariableop_resource:K(]
Oplayer1qnet_player1qnet_encodingnetwork_dense_1_biasadd_readvariableop_resource:(Q
>player1qnet_player1qnet_dense_2_matmul_readvariableop_resource:	(А N
?player1qnet_player1qnet_dense_2_biasadd_readvariableop_resource:	А 
identityИвDPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpвCPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpвFPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpвEPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpв6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOpв5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOpЖ
5Player1QNet/Player1QNet/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   ╟
7Player1QNet/Player1QNet/EncodingNetwork/flatten/ReshapeReshapeobservation_state>Player1QNet/Player1QNet/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         @╜
2Player1QNet/Player1QNet/EncodingNetwork/dense/CastCast@Player1QNet/Player1QNet/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:         @╨
CPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpLplayer1qnet_player1qnet_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:@K*
dtype0ї
4Player1QNet/Player1QNet/EncodingNetwork/dense/MatMulMatMul6Player1QNet/Player1QNet/EncodingNetwork/dense/Cast:y:0KPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         K╬
DPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpMplayer1qnet_player1qnet_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:K*
dtype0А
5Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAddBiasAdd>Player1QNet/Player1QNet/EncodingNetwork/dense/MatMul:product:0LPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Kм
2Player1QNet/Player1QNet/EncodingNetwork/dense/ReluRelu>Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         K╘
EPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNplayer1qnet_player1qnet_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:K(*
dtype0Г
6Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMulMatMul@Player1QNet/Player1QNet/EncodingNetwork/dense/Relu:activations:0MPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (╥
FPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOplayer1qnet_player1qnet_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0Ж
7Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAddBiasAdd@Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul:product:0NPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (░
4Player1QNet/Player1QNet/EncodingNetwork/dense_1/ReluRelu@Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         (╡
5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOpReadVariableOp>player1qnet_player1qnet_dense_2_matmul_readvariableop_resource*
_output_shapes
:	(А *
dtype0ц
&Player1QNet/Player1QNet/dense_2/MatMulMatMulBPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/Relu:activations:0=Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А │
6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOpReadVariableOp?player1qnet_player1qnet_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А *
dtype0╫
'Player1QNet/Player1QNet/dense_2/BiasAddBiasAdd0Player1QNet/Player1QNet/dense_2/MatMul:product:0>Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А q
Player1QNet/ShapeShape0Player1QNet/Player1QNet/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:Y
Player1QNet/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : К
Player1QNet/zerosFillPlayer1QNet/Shape:output:0 Player1QNet/zeros/Const:output:0*
T0*(
_output_shapes
:         А {
Player1QNet/EqualEqualPlayer1QNet/zeros:output:0observation_mask*
T0*(
_output_shapes
:         А {
Player1QNet/AbsAbs0Player1QNet/Player1QNet/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         А [
Player1QNet/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *    а
Player1QNet/SelectV2SelectV2Player1QNet/Equal:z:0Player1QNet/SelectV2/t:output:0Player1QNet/Abs:y:0*
T0*(
_output_shapes
:         А l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         Ъ
Categorical/mode/ArgMaxArgMaxPlayer1QNet/SelectV2:output:0*Categorical/mode/ArgMax/dimension:output:0*
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
valueB:╕
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
valueB о
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
value	B : З
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:в
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
:и
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:         Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B : Ч
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
:         ╒
NoOpNoOpE^Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpD^Player1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpG^Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpF^Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp7^Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp6^Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         :         :         А :         : : : : : : 2М
DPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpDPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOp2К
CPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpCPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOp2Р
FPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpFPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2О
EPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpEPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp2p
6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp2n
5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp:N J
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
:         А 
*
_user_specified_nameobservation/mask:^Z
+
_output_shapes
:         
+
_user_specified_nameobservation/state
└A
щ
0__inference_polymorphic_distribution_fn_66915419
	step_type

reward
discount
observation_mask
observation_state^
Lplayer1qnet_player1qnet_encodingnetwork_dense_matmul_readvariableop_resource:@K[
Mplayer1qnet_player1qnet_encodingnetwork_dense_biasadd_readvariableop_resource:K`
Nplayer1qnet_player1qnet_encodingnetwork_dense_1_matmul_readvariableop_resource:K(]
Oplayer1qnet_player1qnet_encodingnetwork_dense_1_biasadd_readvariableop_resource:(Q
>player1qnet_player1qnet_dense_2_matmul_readvariableop_resource:	(А N
?player1qnet_player1qnet_dense_2_biasadd_readvariableop_resource:	А 
identity

identity_1

identity_2ИвDPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpвCPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpвFPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpвEPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpв6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOpв5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOpЖ
5Player1QNet/Player1QNet/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   ╟
7Player1QNet/Player1QNet/EncodingNetwork/flatten/ReshapeReshapeobservation_state>Player1QNet/Player1QNet/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         @╜
2Player1QNet/Player1QNet/EncodingNetwork/dense/CastCast@Player1QNet/Player1QNet/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:         @╨
CPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpLplayer1qnet_player1qnet_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:@K*
dtype0ї
4Player1QNet/Player1QNet/EncodingNetwork/dense/MatMulMatMul6Player1QNet/Player1QNet/EncodingNetwork/dense/Cast:y:0KPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         K╬
DPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpMplayer1qnet_player1qnet_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:K*
dtype0А
5Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAddBiasAdd>Player1QNet/Player1QNet/EncodingNetwork/dense/MatMul:product:0LPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Kм
2Player1QNet/Player1QNet/EncodingNetwork/dense/ReluRelu>Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         K╘
EPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNplayer1qnet_player1qnet_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:K(*
dtype0Г
6Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMulMatMul@Player1QNet/Player1QNet/EncodingNetwork/dense/Relu:activations:0MPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (╥
FPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOplayer1qnet_player1qnet_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0Ж
7Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAddBiasAdd@Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul:product:0NPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (░
4Player1QNet/Player1QNet/EncodingNetwork/dense_1/ReluRelu@Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         (╡
5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOpReadVariableOp>player1qnet_player1qnet_dense_2_matmul_readvariableop_resource*
_output_shapes
:	(А *
dtype0ц
&Player1QNet/Player1QNet/dense_2/MatMulMatMulBPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/Relu:activations:0=Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А │
6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOpReadVariableOp?player1qnet_player1qnet_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А *
dtype0╫
'Player1QNet/Player1QNet/dense_2/BiasAddBiasAdd0Player1QNet/Player1QNet/dense_2/MatMul:product:0>Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А q
Player1QNet/ShapeShape0Player1QNet/Player1QNet/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:Y
Player1QNet/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : К
Player1QNet/zerosFillPlayer1QNet/Shape:output:0 Player1QNet/zeros/Const:output:0*
T0*(
_output_shapes
:         А {
Player1QNet/EqualEqualPlayer1QNet/zeros:output:0observation_mask*
T0*(
_output_shapes
:         А {
Player1QNet/AbsAbs0Player1QNet/Player1QNet/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         А [
Player1QNet/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *    а
Player1QNet/SelectV2SelectV2Player1QNet/Equal:z:0Player1QNet/SelectV2/t:output:0Player1QNet/Abs:y:0*
T0*(
_output_shapes
:         А l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         Ъ
Categorical/mode/ArgMaxArgMaxPlayer1QNet/SelectV2:output:0*Categorical/mode/ArgMax/dimension:output:0*
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
: ╒
NoOpNoOpE^Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpD^Player1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpG^Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpF^Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp7^Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp6^Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         :         :         А :         : : : : : : 2М
DPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpDPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOp2К
CPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpCPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOp2Р
FPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpFPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2О
EPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpEPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp2p
6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp2n
5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp:N J
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
:         А 
*
_user_specified_nameobservation/mask:^Z
+
_output_shapes
:         
+
_user_specified_nameobservation/state
╟
░
"__inference__traced_save_259972933
file_prefix'
#savev2_variable_read_readvariableop	G
Csavev2_player1qnet_encodingnetwork_dense_kernel_read_readvariableopE
Asavev2_player1qnet_encodingnetwork_dense_bias_read_readvariableopI
Esavev2_player1qnet_encodingnetwork_dense_1_kernel_read_readvariableopG
Csavev2_player1qnet_encodingnetwork_dense_1_bias_read_readvariableop9
5savev2_player1qnet_dense_2_kernel_read_readvariableop7
3savev2_player1qnet_dense_2_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ┼
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ю
valueфBсB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B Д
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopCsavev2_player1qnet_encodingnetwork_dense_kernel_read_readvariableopAsavev2_player1qnet_encodingnetwork_dense_bias_read_readvariableopEsavev2_player1qnet_encodingnetwork_dense_1_kernel_read_readvariableopCsavev2_player1qnet_encodingnetwork_dense_1_bias_read_readvariableop5savev2_player1qnet_dense_2_kernel_read_readvariableop3savev2_player1qnet_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes

2	Р
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
8: : :@K:K:K(:(:	(А :А : 2(
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
:	(А :!

_output_shapes	
:А :

_output_shapes
: 
Ы^
ї
*__inference_polymorphic_action_fn_66915374
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_mask
time_step_observation_state^
Lplayer1qnet_player1qnet_encodingnetwork_dense_matmul_readvariableop_resource:@K[
Mplayer1qnet_player1qnet_encodingnetwork_dense_biasadd_readvariableop_resource:K`
Nplayer1qnet_player1qnet_encodingnetwork_dense_1_matmul_readvariableop_resource:K(]
Oplayer1qnet_player1qnet_encodingnetwork_dense_1_biasadd_readvariableop_resource:(Q
>player1qnet_player1qnet_dense_2_matmul_readvariableop_resource:	(А N
?player1qnet_player1qnet_dense_2_biasadd_readvariableop_resource:	А 
identityИвDPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpвCPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpвFPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpвEPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpв6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOpв5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOpЖ
5Player1QNet/Player1QNet/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   ╤
7Player1QNet/Player1QNet/EncodingNetwork/flatten/ReshapeReshapetime_step_observation_state>Player1QNet/Player1QNet/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         @╜
2Player1QNet/Player1QNet/EncodingNetwork/dense/CastCast@Player1QNet/Player1QNet/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:         @╨
CPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpLplayer1qnet_player1qnet_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:@K*
dtype0ї
4Player1QNet/Player1QNet/EncodingNetwork/dense/MatMulMatMul6Player1QNet/Player1QNet/EncodingNetwork/dense/Cast:y:0KPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         K╬
DPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpMplayer1qnet_player1qnet_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:K*
dtype0А
5Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAddBiasAdd>Player1QNet/Player1QNet/EncodingNetwork/dense/MatMul:product:0LPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Kм
2Player1QNet/Player1QNet/EncodingNetwork/dense/ReluRelu>Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         K╘
EPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNplayer1qnet_player1qnet_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:K(*
dtype0Г
6Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMulMatMul@Player1QNet/Player1QNet/EncodingNetwork/dense/Relu:activations:0MPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (╥
FPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOplayer1qnet_player1qnet_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0Ж
7Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAddBiasAdd@Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul:product:0NPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (░
4Player1QNet/Player1QNet/EncodingNetwork/dense_1/ReluRelu@Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         (╡
5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOpReadVariableOp>player1qnet_player1qnet_dense_2_matmul_readvariableop_resource*
_output_shapes
:	(А *
dtype0ц
&Player1QNet/Player1QNet/dense_2/MatMulMatMulBPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/Relu:activations:0=Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А │
6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOpReadVariableOp?player1qnet_player1qnet_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А *
dtype0╫
'Player1QNet/Player1QNet/dense_2/BiasAddBiasAdd0Player1QNet/Player1QNet/dense_2/MatMul:product:0>Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А q
Player1QNet/ShapeShape0Player1QNet/Player1QNet/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:Y
Player1QNet/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : К
Player1QNet/zerosFillPlayer1QNet/Shape:output:0 Player1QNet/zeros/Const:output:0*
T0*(
_output_shapes
:         А Е
Player1QNet/EqualEqualPlayer1QNet/zeros:output:0time_step_observation_mask*
T0*(
_output_shapes
:         А {
Player1QNet/AbsAbs0Player1QNet/Player1QNet/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         А [
Player1QNet/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *    а
Player1QNet/SelectV2SelectV2Player1QNet/Equal:z:0Player1QNet/SelectV2/t:output:0Player1QNet/Abs:y:0*
T0*(
_output_shapes
:         А l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         Ъ
Categorical/mode/ArgMaxArgMaxPlayer1QNet/SelectV2:output:0*Categorical/mode/ArgMax/dimension:output:0*
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
valueB:╕
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
valueB о
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
value	B : З
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:в
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
:и
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:         Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B : Ч
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
:         ╒
NoOpNoOpE^Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpD^Player1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpG^Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpF^Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp7^Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp6^Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         :         :         А :         : : : : : : 2М
DPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpDPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOp2К
CPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpCPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOp2Р
FPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpFPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2О
EPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpEPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp2p
6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp2n
5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp:X T
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
:         А 
4
_user_specified_nametime_step_observation_mask:hd
+
_output_shapes
:         
5
_user_specified_nametime_step_observation_state
Й]
└
*__inference_polymorphic_action_fn_66915162
	time_step
time_step_1
time_step_2
time_step_3
time_step_4^
Lplayer1qnet_player1qnet_encodingnetwork_dense_matmul_readvariableop_resource:@K[
Mplayer1qnet_player1qnet_encodingnetwork_dense_biasadd_readvariableop_resource:K`
Nplayer1qnet_player1qnet_encodingnetwork_dense_1_matmul_readvariableop_resource:K(]
Oplayer1qnet_player1qnet_encodingnetwork_dense_1_biasadd_readvariableop_resource:(Q
>player1qnet_player1qnet_dense_2_matmul_readvariableop_resource:	(А N
?player1qnet_player1qnet_dense_2_biasadd_readvariableop_resource:	А 
identityИвDPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpвCPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpвFPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpвEPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpв6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOpв5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOpЖ
5Player1QNet/Player1QNet/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   ┴
7Player1QNet/Player1QNet/EncodingNetwork/flatten/ReshapeReshapetime_step_4>Player1QNet/Player1QNet/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         @╜
2Player1QNet/Player1QNet/EncodingNetwork/dense/CastCast@Player1QNet/Player1QNet/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:         @╨
CPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpLplayer1qnet_player1qnet_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:@K*
dtype0ї
4Player1QNet/Player1QNet/EncodingNetwork/dense/MatMulMatMul6Player1QNet/Player1QNet/EncodingNetwork/dense/Cast:y:0KPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         K╬
DPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpMplayer1qnet_player1qnet_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:K*
dtype0А
5Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAddBiasAdd>Player1QNet/Player1QNet/EncodingNetwork/dense/MatMul:product:0LPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Kм
2Player1QNet/Player1QNet/EncodingNetwork/dense/ReluRelu>Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         K╘
EPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNplayer1qnet_player1qnet_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:K(*
dtype0Г
6Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMulMatMul@Player1QNet/Player1QNet/EncodingNetwork/dense/Relu:activations:0MPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (╥
FPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOplayer1qnet_player1qnet_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0Ж
7Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAddBiasAdd@Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul:product:0NPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (░
4Player1QNet/Player1QNet/EncodingNetwork/dense_1/ReluRelu@Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         (╡
5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOpReadVariableOp>player1qnet_player1qnet_dense_2_matmul_readvariableop_resource*
_output_shapes
:	(А *
dtype0ц
&Player1QNet/Player1QNet/dense_2/MatMulMatMulBPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/Relu:activations:0=Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А │
6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOpReadVariableOp?player1qnet_player1qnet_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А *
dtype0╫
'Player1QNet/Player1QNet/dense_2/BiasAddBiasAdd0Player1QNet/Player1QNet/dense_2/MatMul:product:0>Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А q
Player1QNet/ShapeShape0Player1QNet/Player1QNet/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:Y
Player1QNet/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : К
Player1QNet/zerosFillPlayer1QNet/Shape:output:0 Player1QNet/zeros/Const:output:0*
T0*(
_output_shapes
:         А v
Player1QNet/EqualEqualPlayer1QNet/zeros:output:0time_step_3*
T0*(
_output_shapes
:         А {
Player1QNet/AbsAbs0Player1QNet/Player1QNet/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         А [
Player1QNet/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *    а
Player1QNet/SelectV2SelectV2Player1QNet/Equal:z:0Player1QNet/SelectV2/t:output:0Player1QNet/Abs:y:0*
T0*(
_output_shapes
:         А l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         Ъ
Categorical/mode/ArgMaxArgMaxPlayer1QNet/SelectV2:output:0*Categorical/mode/ArgMax/dimension:output:0*
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
valueB:╕
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
valueB о
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
value	B : З
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:в
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
:и
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:         Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B : Ч
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
:         ╒
NoOpNoOpE^Player1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpD^Player1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpG^Player1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpF^Player1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp7^Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp6^Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         :         :         :         А :         : : : : : : 2М
DPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOpDPlayer1QNet/Player1QNet/EncodingNetwork/dense/BiasAdd/ReadVariableOp2К
CPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOpCPlayer1QNet/Player1QNet/EncodingNetwork/dense/MatMul/ReadVariableOp2Р
FPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpFPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2О
EPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOpEPlayer1QNet/Player1QNet/EncodingNetwork/dense_1/MatMul/ReadVariableOp2p
6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp6Player1QNet/Player1QNet/dense_2/BiasAdd/ReadVariableOp2n
5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp5Player1QNet/Player1QNet/dense_2/MatMul/ReadVariableOp:N J
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
:         А 
#
_user_specified_name	time_step:VR
+
_output_shapes
:         
#
_user_specified_name	time_step"Ж
L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ъ
actionП
4

0/discount&
action_0_discount:0         
I
0/observation/mask3
action_0_observation_mask:0         А 
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
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:┼z
ф
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
trace_12М
*__inference_polymorphic_action_fn_66915304
*__inference_polymorphic_action_fn_66915374▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0ztrace_1
Г
trace_02ц
0__inference_polymorphic_distribution_fn_66915419▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0
ю
trace_02╤
&__inference_get_initial_state_66915422ж
Э▓Щ
FullArgSpec!
argsЪ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0
оBл
__inference_<lambda>_617"О
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
оBл
__inference_<lambda>_614"О
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
`

action
get_initial_state
get_train_step
get_metadata"
signature_map
 "
trackable_dict_wrapper
::8@K2(Player1QNet/EncodingNetwork/dense/kernel
4:2K2&Player1QNet/EncodingNetwork/dense/bias
<::K(2*Player1QNet/EncodingNetwork/dense_1/kernel
6:4(2(Player1QNet/EncodingNetwork/dense_1/bias
-:+	(А 2Player1QNet/dense_2/kernel
':%А 2Player1QNet/dense_2/bias
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
зBд
*__inference_polymorphic_action_fn_66915304	step_typerewarddiscountobservation/maskobservation/state"▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘B╓
*__inference_polymorphic_action_fn_66915374time_step_step_typetime_step_rewardtime_step_discounttime_step_observation_masktime_step_observation_state"▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
нBк
0__inference_polymorphic_distribution_fn_66915419	step_typerewarddiscountobservation/maskobservation/state"▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
тB▀
&__inference_get_initial_state_66915422
batch_size"ж
Э▓Щ
FullArgSpec!
argsЪ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ПBМ
'__inference_signature_wrapper_259972866
0/discount0/observation/mask0/observation/state0/reward0/step_type"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤B╬
'__inference_signature_wrapper_259972871
batch_size"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
├B└
'__inference_signature_wrapper_259972879"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
├B└
'__inference_signature_wrapper_259972883"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
н
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
ё2юы
т▓▐
FullArgSpec@
args8Ъ5
jself
jobservation
j	step_type
jnetwork_state
varargs
 
varkw
 
defaultsв

 

 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ё2юы
т▓▐
FullArgSpec@
args8Ъ5
jself
jobservation
j	step_type
jnetwork_state
varargs
 
varkw
 
defaultsв

 

 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
 "
trackable_dict_wrapper
╟
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
н
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
х2т▀
╓▓╥
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв

 
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
х2т▀
╓▓╥
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв

 
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
н
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
х2т▀
╓▓╥
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв

 
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
х2т▀
╓▓╥
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв

 
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
н
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
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
е
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
н
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
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
н
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
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
н
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
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
__inference_<lambda>_614$в

в 
к "К
unknown 	0
__inference_<lambda>_617в

в 
к "к S
&__inference_get_initial_state_66915422)"в
в
К

batch_size 
к "в ║
*__inference_polymorphic_action_fn_66915304Лмви
авЬ
Ф▓Р
TimeStep,
	step_typeК
	step_type         &
rewardК
reward         *
discountК
discount         Б
observationrкo
3
mask+К(
observation/mask         А 
8
state/К,
observation/state         
в 
к "R▓O

PolicyStep&
actionК
action         
stateв 
infoв ю
*__inference_polymorphic_action_fn_66915374┐рв▄
╘в╨
╚▓─
TimeStep6
	step_type)К&
time_step_step_type         0
reward&К#
time_step_reward         4
discount(К%
time_step_discount         Ч
observationЗкГ
=
mask5К2
time_step_observation_mask         А 
B
state9К6
time_step_observation_state         
в 
к "R▓O

PolicyStep&
actionК
action         
stateв 
infoв Я
0__inference_polymorphic_distribution_fn_66915419ъмви
авЬ
Ф▓Р
TimeStep,
	step_typeК
	step_type         &
rewardК
reward         *
discountК
discount         Б
observationrкo
3
mask+К(
observation/mask         А 
8
state/К,
observation/state         
в 
к "░▓м

PolicyStepВ
actionўТє┐в╗
`
Bк?

atolК 

locК         

rtolК 
LкI

allow_nan_statsp

namejDeterministic_1_1

validate_argsp 
в
j
parameters
в 
в
jname+tfp.distributions.Deterministic_ACTTypeSpec 
stateв 
infoв ┤
'__inference_signature_wrapper_259972866И╨в╠
в 
─к└
5

0/discount'К$
tensor_0_discount         
J
0/observation/mask4К1
tensor_0_observation_mask         А 
O
0/observation/state8К5
tensor_0_observation_state         
1
0/reward%К"
tensor_0_reward         
7
0/step_type(К%
tensor_0_step_type         "+к(
&
actionК
action         b
'__inference_signature_wrapper_25997287170в-
в 
&к#
!

batch_sizeК

batch_size "к [
'__inference_signature_wrapper_2599728790в

в 
к "к

int64К
int64 	?
'__inference_signature_wrapper_259972883в

в 
к "к 