??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
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
dtypetype?
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
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
string_lookup_1_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name
table_15*
value_dtype0
?
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_2*
value_dtype0	
?
$query_generator/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*5
shared_name&$query_generator/embedding/embeddings
?
8query_generator/embedding/embeddings/Read/ReadVariableOpReadVariableOp$query_generator/embedding/embeddings*
_output_shapes
:	?d*
dtype0
?
query_generator/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_namequery_generator/dense/kernel
?
0query_generator/dense/kernel/Read/ReadVariableOpReadVariableOpquery_generator/dense/kernel* 
_output_shapes
:
??*
dtype0
?
query_generator/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namequery_generator/dense/bias
?
.query_generator/dense/bias/Read/ReadVariableOpReadVariableOpquery_generator/dense/bias*
_output_shapes	
:?*
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
?
#query_generator/gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*4
shared_name%#query_generator/gru/gru_cell/kernel
?
7query_generator/gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOp#query_generator/gru/gru_cell/kernel*
_output_shapes
:	d?*
dtype0
?
-query_generator/gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*>
shared_name/-query_generator/gru/gru_cell/recurrent_kernel
?
Aquery_generator/gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp-query_generator/gru/gru_cell/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
!query_generator/gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!query_generator/gru/gru_cell/bias
?
5query_generator/gru/gru_cell/bias/Read/ReadVariableOpReadVariableOp!query_generator/gru/gru_cell/bias*
_output_shapes
:	?*
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
?
+Adam/query_generator/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*<
shared_name-+Adam/query_generator/embedding/embeddings/m
?
?Adam/query_generator/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp+Adam/query_generator/embedding/embeddings/m*
_output_shapes
:	?d*
dtype0
?
#Adam/query_generator/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adam/query_generator/dense/kernel/m
?
7Adam/query_generator/dense/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/query_generator/dense/kernel/m* 
_output_shapes
:
??*
dtype0
?
!Adam/query_generator/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/query_generator/dense/bias/m
?
5Adam/query_generator/dense/bias/m/Read/ReadVariableOpReadVariableOp!Adam/query_generator/dense/bias/m*
_output_shapes	
:?*
dtype0
?
*Adam/query_generator/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*;
shared_name,*Adam/query_generator/gru/gru_cell/kernel/m
?
>Adam/query_generator/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/query_generator/gru/gru_cell/kernel/m*
_output_shapes
:	d?*
dtype0
?
4Adam/query_generator/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*E
shared_name64Adam/query_generator/gru/gru_cell/recurrent_kernel/m
?
HAdam/query_generator/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp4Adam/query_generator/gru/gru_cell/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
(Adam/query_generator/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*9
shared_name*(Adam/query_generator/gru/gru_cell/bias/m
?
<Adam/query_generator/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOp(Adam/query_generator/gru/gru_cell/bias/m*
_output_shapes
:	?*
dtype0
?
+Adam/query_generator/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*<
shared_name-+Adam/query_generator/embedding/embeddings/v
?
?Adam/query_generator/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp+Adam/query_generator/embedding/embeddings/v*
_output_shapes
:	?d*
dtype0
?
#Adam/query_generator/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#Adam/query_generator/dense/kernel/v
?
7Adam/query_generator/dense/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/query_generator/dense/kernel/v* 
_output_shapes
:
??*
dtype0
?
!Adam/query_generator/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/query_generator/dense/bias/v
?
5Adam/query_generator/dense/bias/v/Read/ReadVariableOpReadVariableOp!Adam/query_generator/dense/bias/v*
_output_shapes	
:?*
dtype0
?
*Adam/query_generator/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*;
shared_name,*Adam/query_generator/gru/gru_cell/kernel/v
?
>Adam/query_generator/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/query_generator/gru/gru_cell/kernel/v*
_output_shapes
:	d?*
dtype0
?
4Adam/query_generator/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*E
shared_name64Adam/query_generator/gru/gru_cell/recurrent_kernel/v
?
HAdam/query_generator/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp4Adam/query_generator/gru/gru_cell/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
(Adam/query_generator/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*9
shared_name*(Adam/query_generator/gru/gru_cell/bias/v
?
<Adam/query_generator/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOp(Adam/query_generator/gru/gru_cell/bias/v*
_output_shapes
:	?*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_1Const*
_output_shapes	
:?*
dtype0*?
value?B??"?  ??                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
M
Const_2Const*
_output_shapes
: *
dtype0*
valueB B[UNK]
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_11483
?
PartitionedCall_1PartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_11488
2
NoOpNoOp^PartitionedCall^PartitionedCall_1
?
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_1_index_table*
Tkeys0	*
Tvalues0*.
_class$
" loc:@string_lookup_1_index_table*
_output_shapes

::
?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_index_table*
Tkeys0*
Tvalues0	*,
_class"
 loc:@string_lookup_index_table*
_output_shapes

::
?*
Const_3Const"/device:CPU:0*
_output_shapes
: *
dtype0*?)
value?)B?) B?)
R
	model
chars_from_ids
ids_from_chars
	keras_api

signatures
?
	embedding
rnn
	dense
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api
0
state_variables

_table
	keras_api
0
state_variables

_table
	keras_api
 
 
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
?
%iter

&beta_1

'beta_2
	(decay
)learning_ratemPmQ mR*mS+mT,mUvVvW vX*vY+vZ,v[
 
*
0
*1
+2
,3
4
 5
*
0
*1
+2
,3
4
 5
?

regularization_losses
	variables

-layers
.non_trainable_variables
/metrics
0layer_regularization_losses
1layer_metrics
trainable_variables
 
20
table'chars_from_ids/_table/.ATTRIBUTES/table
 
 
20
table'ids_from_chars/_table/.ATTRIBUTES/table
 
om
VARIABLE_VALUE$query_generator/embedding/embeddings5model/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
regularization_losses
	variables

2layers
3non_trainable_variables
4metrics
5layer_regularization_losses
6layer_metrics
trainable_variables
~

*kernel
+recurrent_kernel
,bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
 
 

*0
+1
,2

*0
+1
,2
?
regularization_losses
	variables

;states

<layers
=non_trainable_variables
>metrics
?layer_regularization_losses
@layer_metrics
trainable_variables
_]
VARIABLE_VALUEquery_generator/dense/kernel-model/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEquery_generator/dense/bias+model/dense/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
?
!regularization_losses
"	variables

Alayers
Bnon_trainable_variables
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
#trainable_variables
NL
VARIABLE_VALUE	Adam/iter/model/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEAdam/beta_11model/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEAdam/beta_21model/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
Adam/decay0model/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEAdam/learning_rate8model/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#query_generator/gru/gru_cell/kernel,model/variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE-query_generator/gru/gru_cell/recurrent_kernel,model/variables/2/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE!query_generator/gru/gru_cell/bias,model/variables/3/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

F0
 
 
 
 
 
 
 
 

*0
+1
,2

*0
+1
,2
?
7regularization_losses
8	variables

Glayers
Hnon_trainable_variables
Imetrics
Jlayer_regularization_losses
Klayer_metrics
9trainable_variables
 

0
 
 
 
 
 
 
 
 
 
4
	Ltotal
	Mcount
N	variables
O	keras_api
 
 
 
 
 
US
VARIABLE_VALUEtotal:model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcount:model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

N	variables
??
VARIABLE_VALUE+Adam/query_generator/embedding/embeddings/mWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/query_generator/dense/kernel/mOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/query_generator/dense/bias/mMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/query_generator/gru/gru_cell/kernel/mNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/query_generator/gru/gru_cell/recurrent_kernel/mNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/query_generator/gru/gru_cell/bias/mNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/query_generator/embedding/embeddings/vWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/query_generator/dense/kernel/vOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/query_generator/dense/bias/vMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/query_generator/gru/gru_cell/kernel/vNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/query_generator/gru/gru_cell/recurrent_kernel/vNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/query_generator/gru/gru_cell/bias/vNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameJstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:1Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:18query_generator/embedding/embeddings/Read/ReadVariableOp0query_generator/dense/kernel/Read/ReadVariableOp.query_generator/dense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp7query_generator/gru/gru_cell/kernel/Read/ReadVariableOpAquery_generator/gru/gru_cell/recurrent_kernel/Read/ReadVariableOp5query_generator/gru/gru_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp?Adam/query_generator/embedding/embeddings/m/Read/ReadVariableOp7Adam/query_generator/dense/kernel/m/Read/ReadVariableOp5Adam/query_generator/dense/bias/m/Read/ReadVariableOp>Adam/query_generator/gru/gru_cell/kernel/m/Read/ReadVariableOpHAdam/query_generator/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOp<Adam/query_generator/gru/gru_cell/bias/m/Read/ReadVariableOp?Adam/query_generator/embedding/embeddings/v/Read/ReadVariableOp7Adam/query_generator/dense/kernel/v/Read/ReadVariableOp5Adam/query_generator/dense/bias/v/Read/ReadVariableOp>Adam/query_generator/gru/gru_cell/kernel/v/Read/ReadVariableOpHAdam/query_generator/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOp<Adam/query_generator/gru/gru_cell/bias/v/Read/ReadVariableOpConst_3**
Tin#
!2			*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_11599
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamestring_lookup_1_index_tablestring_lookup_index_table$query_generator/embedding/embeddingsquery_generator/dense/kernelquery_generator/dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate#query_generator/gru/gru_cell/kernel-query_generator/gru/gru_cell/recurrent_kernel!query_generator/gru/gru_cell/biastotalcount+Adam/query_generator/embedding/embeddings/m#Adam/query_generator/dense/kernel/m!Adam/query_generator/dense/bias/m*Adam/query_generator/gru/gru_cell/kernel/m4Adam/query_generator/gru/gru_cell/recurrent_kernel/m(Adam/query_generator/gru/gru_cell/bias/m+Adam/query_generator/embedding/embeddings/v#Adam/query_generator/dense/kernel/v!Adam/query_generator/dense/bias/v*Adam/query_generator/gru/gru_cell/kernel/v4Adam/query_generator/gru/gru_cell/recurrent_kernel/v(Adam/query_generator/gru/gru_cell/bias/v*'
Tin 
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_11690??
? 
?
?__inference_dense_layer_call_and_return_conditional_losses_8877

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????K?2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????K?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????K?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????K?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????K?
 
_user_specified_nameinputs
?
,
__inference__destroyer_11303
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?K
?
gru_while_body_9824$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2!
gru_while_gru_strided_slice_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	?F
3gru_while_gru_cell_matmul_readvariableop_resource_0:	d?I
5gru_while_gru_cell_matmul_1_readvariableop_resource_0:
??
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_gru_strided_slice]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	?D
1gru_while_gru_cell_matmul_readvariableop_resource:	d?G
3gru_while_gru_cell_matmul_1_readvariableop_resource:
????(gru/while/gru_cell/MatMul/ReadVariableOp?*gru/while/gru_cell/MatMul_1/ReadVariableOp?!gru/while/gru_cell/ReadVariableOp?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/while/gru_cell/unstack?
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(gru/while/gru_cell/MatMul/ReadVariableOp?
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd?
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru/while/gru_cell/split/split_dim?
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/while/gru_cell/split?
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*gru/while/gru_cell/MatMul_1/ReadVariableOp?
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_1?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_1?
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru/while/gru_cell/Const?
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru/while/gru_cell/split_1/split_dim?
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/while/gru_cell/split_1?
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add?
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid?
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_1?
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid_1?
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul?
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_2?
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Relu?
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_1y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_2?
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_3?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1?
gru/while/IdentityIdentitygru/while/add_1:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_1?
gru/while/Identity_2Identitygru/while/add:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
gru/while/Identity_4"l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"<
gru_while_gru_strided_slicegru_while_gru_strided_slice_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?c
?
#query_generator_gru_while_body_9615D
@query_generator_gru_while_query_generator_gru_while_loop_counterJ
Fquery_generator_gru_while_query_generator_gru_while_maximum_iterations)
%query_generator_gru_while_placeholder+
'query_generator_gru_while_placeholder_1+
'query_generator_gru_while_placeholder_2A
=query_generator_gru_while_query_generator_gru_strided_slice_0
{query_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensor_0O
<query_generator_gru_while_gru_cell_readvariableop_resource_0:	?V
Cquery_generator_gru_while_gru_cell_matmul_readvariableop_resource_0:	d?Y
Equery_generator_gru_while_gru_cell_matmul_1_readvariableop_resource_0:
??&
"query_generator_gru_while_identity(
$query_generator_gru_while_identity_1(
$query_generator_gru_while_identity_2(
$query_generator_gru_while_identity_3(
$query_generator_gru_while_identity_4?
;query_generator_gru_while_query_generator_gru_strided_slice}
yquery_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensorM
:query_generator_gru_while_gru_cell_readvariableop_resource:	?T
Aquery_generator_gru_while_gru_cell_matmul_readvariableop_resource:	d?W
Cquery_generator_gru_while_gru_cell_matmul_1_readvariableop_resource:
????8query_generator/gru/while/gru_cell/MatMul/ReadVariableOp?:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp?1query_generator/gru/while/gru_cell/ReadVariableOp?
Kquery_generator/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   d   2M
Kquery_generator/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=query_generator/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{query_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensor_0%query_generator_gru_while_placeholderTquery_generator/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:d*
element_dtype02?
=query_generator/gru/while/TensorArrayV2Read/TensorListGetItem?
1query_generator/gru/while/gru_cell/ReadVariableOpReadVariableOp<query_generator_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype023
1query_generator/gru/while/gru_cell/ReadVariableOp?
*query_generator/gru/while/gru_cell/unstackUnpack9query_generator/gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2,
*query_generator/gru/while/gru_cell/unstack?
8query_generator/gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOpCquery_generator_gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02:
8query_generator/gru/while/gru_cell/MatMul/ReadVariableOp?
)query_generator/gru/while/gru_cell/MatMulMatMulDquery_generator/gru/while/TensorArrayV2Read/TensorListGetItem:item:0@query_generator/gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2+
)query_generator/gru/while/gru_cell/MatMul?
*query_generator/gru/while/gru_cell/BiasAddBiasAdd3query_generator/gru/while/gru_cell/MatMul:product:03query_generator/gru/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	?2,
*query_generator/gru/while/gru_cell/BiasAdd?
2query_generator/gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2query_generator/gru/while/gru_cell/split/split_dim?
(query_generator/gru/while/gru_cell/splitSplit;query_generator/gru/while/gru_cell/split/split_dim:output:03query_generator/gru/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_split2*
(query_generator/gru/while/gru_cell/split?
:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpEquery_generator_gru_while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02<
:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp?
+query_generator/gru/while/gru_cell/MatMul_1MatMul'query_generator_gru_while_placeholder_2Bquery_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2-
+query_generator/gru/while/gru_cell/MatMul_1?
,query_generator/gru/while/gru_cell/BiasAdd_1BiasAdd5query_generator/gru/while/gru_cell/MatMul_1:product:03query_generator/gru/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	?2.
,query_generator/gru/while/gru_cell/BiasAdd_1?
(query_generator/gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2*
(query_generator/gru/while/gru_cell/Const?
4query_generator/gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4query_generator/gru/while/gru_cell/split_1/split_dim?
*query_generator/gru/while/gru_cell/split_1SplitV5query_generator/gru/while/gru_cell/BiasAdd_1:output:01query_generator/gru/while/gru_cell/Const:output:0=query_generator/gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	?:	?:	?*
	num_split2,
*query_generator/gru/while/gru_cell/split_1?
&query_generator/gru/while/gru_cell/addAddV21query_generator/gru/while/gru_cell/split:output:03query_generator/gru/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	?2(
&query_generator/gru/while/gru_cell/add?
*query_generator/gru/while/gru_cell/SigmoidSigmoid*query_generator/gru/while/gru_cell/add:z:0*
T0*
_output_shapes
:	?2,
*query_generator/gru/while/gru_cell/Sigmoid?
(query_generator/gru/while/gru_cell/add_1AddV21query_generator/gru/while/gru_cell/split:output:13query_generator/gru/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	?2*
(query_generator/gru/while/gru_cell/add_1?
,query_generator/gru/while/gru_cell/Sigmoid_1Sigmoid,query_generator/gru/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	?2.
,query_generator/gru/while/gru_cell/Sigmoid_1?
&query_generator/gru/while/gru_cell/mulMul0query_generator/gru/while/gru_cell/Sigmoid_1:y:03query_generator/gru/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	?2(
&query_generator/gru/while/gru_cell/mul?
(query_generator/gru/while/gru_cell/add_2AddV21query_generator/gru/while/gru_cell/split:output:2*query_generator/gru/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	?2*
(query_generator/gru/while/gru_cell/add_2?
'query_generator/gru/while/gru_cell/ReluRelu,query_generator/gru/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	?2)
'query_generator/gru/while/gru_cell/Relu?
(query_generator/gru/while/gru_cell/mul_1Mul.query_generator/gru/while/gru_cell/Sigmoid:y:0'query_generator_gru_while_placeholder_2*
T0*
_output_shapes
:	?2*
(query_generator/gru/while/gru_cell/mul_1?
(query_generator/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(query_generator/gru/while/gru_cell/sub/x?
&query_generator/gru/while/gru_cell/subSub1query_generator/gru/while/gru_cell/sub/x:output:0.query_generator/gru/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	?2(
&query_generator/gru/while/gru_cell/sub?
(query_generator/gru/while/gru_cell/mul_2Mul*query_generator/gru/while/gru_cell/sub:z:05query_generator/gru/while/gru_cell/Relu:activations:0*
T0*
_output_shapes
:	?2*
(query_generator/gru/while/gru_cell/mul_2?
(query_generator/gru/while/gru_cell/add_3AddV2,query_generator/gru/while/gru_cell/mul_1:z:0,query_generator/gru/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	?2*
(query_generator/gru/while/gru_cell/add_3?
>query_generator/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'query_generator_gru_while_placeholder_1%query_generator_gru_while_placeholder,query_generator/gru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02@
>query_generator/gru/while/TensorArrayV2Write/TensorListSetItem?
query_generator/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
query_generator/gru/while/add/y?
query_generator/gru/while/addAddV2%query_generator_gru_while_placeholder(query_generator/gru/while/add/y:output:0*
T0*
_output_shapes
: 2
query_generator/gru/while/add?
!query_generator/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!query_generator/gru/while/add_1/y?
query_generator/gru/while/add_1AddV2@query_generator_gru_while_query_generator_gru_while_loop_counter*query_generator/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
query_generator/gru/while/add_1?
"query_generator/gru/while/IdentityIdentity#query_generator/gru/while/add_1:z:09^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2$
"query_generator/gru/while/Identity?
$query_generator/gru/while/Identity_1IdentityFquery_generator_gru_while_query_generator_gru_while_maximum_iterations9^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2&
$query_generator/gru/while/Identity_1?
$query_generator/gru/while/Identity_2Identity!query_generator/gru/while/add:z:09^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2&
$query_generator/gru/while/Identity_2?
$query_generator/gru/while/Identity_3IdentityNquery_generator/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:09^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2&
$query_generator/gru/while/Identity_3?
$query_generator/gru/while/Identity_4Identity,query_generator/gru/while/gru_cell/add_3:z:09^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
:	?2&
$query_generator/gru/while/Identity_4"?
Cquery_generator_gru_while_gru_cell_matmul_1_readvariableop_resourceEquery_generator_gru_while_gru_cell_matmul_1_readvariableop_resource_0"?
Aquery_generator_gru_while_gru_cell_matmul_readvariableop_resourceCquery_generator_gru_while_gru_cell_matmul_readvariableop_resource_0"z
:query_generator_gru_while_gru_cell_readvariableop_resource<query_generator_gru_while_gru_cell_readvariableop_resource_0"Q
"query_generator_gru_while_identity+query_generator/gru/while/Identity:output:0"U
$query_generator_gru_while_identity_1-query_generator/gru/while/Identity_1:output:0"U
$query_generator_gru_while_identity_2-query_generator/gru/while/Identity_2:output:0"U
$query_generator_gru_while_identity_3-query_generator/gru/while/Identity_3:output:0"U
$query_generator_gru_while_identity_4-query_generator/gru/while/Identity_4:output:0"|
;query_generator_gru_while_query_generator_gru_strided_slice=query_generator_gru_while_query_generator_gru_strided_slice_0"?
yquery_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensor{query_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	?: : : : : 2t
8query_generator/gru/while/gru_cell/MatMul/ReadVariableOp8query_generator/gru/while/gru_cell/MatMul/ReadVariableOp2x
:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2f
1query_generator/gru/while/gru_cell/ReadVariableOp1query_generator/gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?K
?
gru_while_body_10012$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2!
gru_while_gru_strided_slice_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	?F
3gru_while_gru_cell_matmul_readvariableop_resource_0:	d?I
5gru_while_gru_cell_matmul_1_readvariableop_resource_0:
??
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_gru_strided_slice]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	?D
1gru_while_gru_cell_matmul_readvariableop_resource:	d?G
3gru_while_gru_cell_matmul_1_readvariableop_resource:
????(gru/while/gru_cell/MatMul/ReadVariableOp?*gru/while/gru_cell/MatMul_1/ReadVariableOp?!gru/while/gru_cell/ReadVariableOp?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/while/gru_cell/unstack?
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(gru/while/gru_cell/MatMul/ReadVariableOp?
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd?
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru/while/gru_cell/split/split_dim?
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/while/gru_cell/split?
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*gru/while/gru_cell/MatMul_1/ReadVariableOp?
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_1?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_1?
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru/while/gru_cell/Const?
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru/while/gru_cell/split_1/split_dim?
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/while/gru_cell/split_1?
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add?
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid?
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_1?
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid_1?
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul?
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_2?
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Relu?
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_1y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_2?
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_3?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1?
gru/while/IdentityIdentitygru/while/add_1:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_1?
gru/while/Identity_2Identitygru/while/add:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
gru/while/Identity_4"l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"<
gru_while_gru_strided_slicegru_while_gru_strided_slice_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_gru_cell_layer_call_and_return_conditional_losses_8286

inputs

states*
readvariableop_resource:	?1
matmul_readvariableop_resource:	d?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3?
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?K
?
gru_while_body_10200$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2!
gru_while_gru_strided_slice_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	?F
3gru_while_gru_cell_matmul_readvariableop_resource_0:	d?I
5gru_while_gru_cell_matmul_1_readvariableop_resource_0:
??
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_gru_strided_slice]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	?D
1gru_while_gru_cell_matmul_readvariableop_resource:	d?G
3gru_while_gru_cell_matmul_1_readvariableop_resource:
????(gru/while/gru_cell/MatMul/ReadVariableOp?*gru/while/gru_cell/MatMul_1/ReadVariableOp?!gru/while/gru_cell/ReadVariableOp?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/while/gru_cell/unstack?
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(gru/while/gru_cell/MatMul/ReadVariableOp?
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd?
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru/while/gru_cell/split/split_dim?
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/while/gru_cell/split?
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*gru/while/gru_cell/MatMul_1/ReadVariableOp?
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_1?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_1?
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru/while/gru_cell/Const?
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru/while/gru_cell/split_1/split_dim?
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/while/gru_cell/split_1?
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add?
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid?
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_1?
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid_1?
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul?
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_2?
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Relu?
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_1y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_2?
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_3?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1?
gru/while/IdentityIdentitygru/while/add_1:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_1?
gru/while/Identity_2Identitygru/while/add:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
gru/while/Identity_4"l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"<
gru_while_gru_strided_slicegru_while_gru_strided_slice_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?

"__inference_generate_one_step_9499

inputsG
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	B
/query_generator_embedding_embedding_lookup_9299:	?dG
4query_generator_gru_gru_cell_readvariableop_resource:	?N
;query_generator_gru_gru_cell_matmul_readvariableop_resource:	d?Q
=query_generator_gru_gru_cell_matmul_1_readvariableop_resource:
??K
7query_generator_dense_tensordot_readvariableop_resource:
??D
5query_generator_dense_biasadd_readvariableop_resource:	?	
add_yI
Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value
identity

identity_1??,query_generator/dense/BiasAdd/ReadVariableOp?.query_generator/dense/Tensordot/ReadVariableOp?*query_generator/embedding/embedding_lookup?2query_generator/gru/gru_cell/MatMul/ReadVariableOp?4query_generator/gru/gru_cell/MatMul_1/ReadVariableOp?+query_generator/gru/gru_cell/ReadVariableOp?query_generator/gru/while?6string_lookup/None_lookup_table_find/LookupTableFindV2?8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
UnicodeSplit/Reshape/shape?
UnicodeSplit/ReshapeReshapeinputs#UnicodeSplit/Reshape/shape:output:0*
T0*
_output_shapes
:2
UnicodeSplit/Reshape?
UnicodeSplit/UnicodeDecodeUnicodeDecodeUnicodeSplit/Reshape:output:0*)
_output_shapes
::?????????*
input_encodingUTF-82
UnicodeSplit/UnicodeDecode?
,UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,UnicodeSplit/RaggedExpandDims/ExpandDims/dim?
(UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims(UnicodeSplit/UnicodeDecode:char_values:05UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2*
(UnicodeSplit/RaggedExpandDims/ExpandDims?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2A
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape?
MUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
MUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2?
GUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0VUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2I
GUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2?
=UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMulRUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: 2?
=UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_mask2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackAUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0?
EUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
EUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis?
@UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0NUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:2B
@UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat?
AUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:?????????2C
AUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R2A
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const?
ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShapeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	2\
ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape?
hUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2j
hUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack?
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2l
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1?
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2l
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2?
bUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicecUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0qUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2d
bUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice?
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2}
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y?
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: 2{
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/CastCast?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1Cast?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1?
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRange?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast:y:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2}
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range?
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMul?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0HUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:?????????2{
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul?
DUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncodeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:?????????*
output_encodingUTF-82F
DUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleMUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2`
string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2!
string_lookup/assert_equal/NoOp?
string_lookup/IdentityIdentity?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
string_lookup/Identity_1Identity'UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
_output_shapes
:2
string_lookup/Identity_1n
RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
RaggedToTensor/zerosw
RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
RaggedToTensor/Const?
#RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorRaggedToTensor/Const:output:0string_lookup/Identity:output:0RaggedToTensor/zeros:output:0!string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2%
#RaggedToTensor/RaggedTensorToTensor?
*query_generator/embedding/embedding_lookupResourceGather/query_generator_embedding_embedding_lookup_9299,RaggedToTensor/RaggedTensorToTensor:result:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*B
_class8
64loc:@query_generator/embedding/embedding_lookup/9299*+
_output_shapes
:?????????d*
dtype02,
*query_generator/embedding/embedding_lookup?
3query_generator/embedding/embedding_lookup/IdentityIdentity3query_generator/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@query_generator/embedding/embedding_lookup/9299*+
_output_shapes
:?????????d25
3query_generator/embedding/embedding_lookup/Identity?
5query_generator/embedding/embedding_lookup/Identity_1Identity<query_generator/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d27
5query_generator/embedding/embedding_lookup/Identity_1?
query_generator/ShapeShape>query_generator/embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
query_generator/Shape?
#query_generator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#query_generator/strided_slice/stack?
%query_generator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%query_generator/strided_slice/stack_1?
%query_generator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%query_generator/strided_slice/stack_2?
query_generator/strided_sliceStridedSlicequery_generator/Shape:output:0,query_generator/strided_slice/stack:output:0.query_generator/strided_slice/stack_1:output:0.query_generator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
query_generator/strided_slice}
query_generator/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
query_generator/zeros/mul/y?
query_generator/zeros/mulMul&query_generator/strided_slice:output:0$query_generator/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
query_generator/zeros/mul
query_generator/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
query_generator/zeros/Less/y?
query_generator/zeros/LessLessquery_generator/zeros/mul:z:0%query_generator/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
query_generator/zeros/Less?
query_generator/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2 
query_generator/zeros/packed/1?
query_generator/zeros/packedPack&query_generator/strided_slice:output:0'query_generator/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
query_generator/zeros/packed
query_generator/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
query_generator/zeros/Const?
query_generator/zerosFill%query_generator/zeros/packed:output:0$query_generator/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
query_generator/zeros?
"query_generator/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"query_generator/gru/transpose/perm?
query_generator/gru/transpose	Transpose>query_generator/embedding/embedding_lookup/Identity_1:output:0+query_generator/gru/transpose/perm:output:0*
T0*+
_output_shapes
:?????????d2
query_generator/gru/transpose?
query_generator/gru/ShapeShape!query_generator/gru/transpose:y:0*
T0*
_output_shapes
:2
query_generator/gru/Shape?
'query_generator/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'query_generator/gru/strided_slice/stack?
)query_generator/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)query_generator/gru/strided_slice/stack_1?
)query_generator/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)query_generator/gru/strided_slice/stack_2?
!query_generator/gru/strided_sliceStridedSlice"query_generator/gru/Shape:output:00query_generator/gru/strided_slice/stack:output:02query_generator/gru/strided_slice/stack_1:output:02query_generator/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!query_generator/gru/strided_slice?
/query_generator/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/query_generator/gru/TensorArrayV2/element_shape?
!query_generator/gru/TensorArrayV2TensorListReserve8query_generator/gru/TensorArrayV2/element_shape:output:0*query_generator/gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!query_generator/gru/TensorArrayV2?
Iquery_generator/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   d   2K
Iquery_generator/gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
;query_generator/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!query_generator/gru/transpose:y:0Rquery_generator/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;query_generator/gru/TensorArrayUnstack/TensorListFromTensor?
)query_generator/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)query_generator/gru/strided_slice_1/stack?
+query_generator/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+query_generator/gru/strided_slice_1/stack_1?
+query_generator/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+query_generator/gru/strided_slice_1/stack_2?
#query_generator/gru/strided_slice_1StridedSlice!query_generator/gru/transpose:y:02query_generator/gru/strided_slice_1/stack:output:04query_generator/gru/strided_slice_1/stack_1:output:04query_generator/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_mask2%
#query_generator/gru/strided_slice_1?
+query_generator/gru/gru_cell/ReadVariableOpReadVariableOp4query_generator_gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+query_generator/gru/gru_cell/ReadVariableOp?
$query_generator/gru/gru_cell/unstackUnpack3query_generator/gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2&
$query_generator/gru/gru_cell/unstack?
2query_generator/gru/gru_cell/MatMul/ReadVariableOpReadVariableOp;query_generator_gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype024
2query_generator/gru/gru_cell/MatMul/ReadVariableOp?
#query_generator/gru/gru_cell/MatMulMatMul,query_generator/gru/strided_slice_1:output:0:query_generator/gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2%
#query_generator/gru/gru_cell/MatMul?
$query_generator/gru/gru_cell/BiasAddBiasAdd-query_generator/gru/gru_cell/MatMul:product:0-query_generator/gru/gru_cell/unstack:output:0*
T0*
_output_shapes
:	?2&
$query_generator/gru/gru_cell/BiasAdd?
,query_generator/gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,query_generator/gru/gru_cell/split/split_dim?
"query_generator/gru/gru_cell/splitSplit5query_generator/gru/gru_cell/split/split_dim:output:0-query_generator/gru/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_split2$
"query_generator/gru/gru_cell/split?
4query_generator/gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp=query_generator_gru_gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4query_generator/gru/gru_cell/MatMul_1/ReadVariableOp?
%query_generator/gru/gru_cell/MatMul_1MatMulquery_generator/zeros:output:0<query_generator/gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%query_generator/gru/gru_cell/MatMul_1?
&query_generator/gru/gru_cell/BiasAdd_1BiasAdd/query_generator/gru/gru_cell/MatMul_1:product:0-query_generator/gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2(
&query_generator/gru/gru_cell/BiasAdd_1?
"query_generator/gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2$
"query_generator/gru/gru_cell/Const?
.query_generator/gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.query_generator/gru/gru_cell/split_1/split_dim?
$query_generator/gru/gru_cell/split_1SplitV/query_generator/gru/gru_cell/BiasAdd_1:output:0+query_generator/gru/gru_cell/Const:output:07query_generator/gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2&
$query_generator/gru/gru_cell/split_1?
 query_generator/gru/gru_cell/addAddV2+query_generator/gru/gru_cell/split:output:0-query_generator/gru/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2"
 query_generator/gru/gru_cell/add?
$query_generator/gru/gru_cell/SigmoidSigmoid$query_generator/gru/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2&
$query_generator/gru/gru_cell/Sigmoid?
"query_generator/gru/gru_cell/add_1AddV2+query_generator/gru/gru_cell/split:output:1-query_generator/gru/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2$
"query_generator/gru/gru_cell/add_1?
&query_generator/gru/gru_cell/Sigmoid_1Sigmoid&query_generator/gru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2(
&query_generator/gru/gru_cell/Sigmoid_1?
 query_generator/gru/gru_cell/mulMul*query_generator/gru/gru_cell/Sigmoid_1:y:0-query_generator/gru/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 query_generator/gru/gru_cell/mul?
"query_generator/gru/gru_cell/add_2AddV2+query_generator/gru/gru_cell/split:output:2$query_generator/gru/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2$
"query_generator/gru/gru_cell/add_2?
!query_generator/gru/gru_cell/ReluRelu&query_generator/gru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2#
!query_generator/gru/gru_cell/Relu?
"query_generator/gru/gru_cell/mul_1Mul(query_generator/gru/gru_cell/Sigmoid:y:0query_generator/zeros:output:0*
T0*(
_output_shapes
:??????????2$
"query_generator/gru/gru_cell/mul_1?
"query_generator/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"query_generator/gru/gru_cell/sub/x?
 query_generator/gru/gru_cell/subSub+query_generator/gru/gru_cell/sub/x:output:0(query_generator/gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2"
 query_generator/gru/gru_cell/sub?
"query_generator/gru/gru_cell/mul_2Mul$query_generator/gru/gru_cell/sub:z:0/query_generator/gru/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2$
"query_generator/gru/gru_cell/mul_2?
"query_generator/gru/gru_cell/add_3AddV2&query_generator/gru/gru_cell/mul_1:z:0&query_generator/gru/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2$
"query_generator/gru/gru_cell/add_3?
1query_generator/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   23
1query_generator/gru/TensorArrayV2_1/element_shape?
#query_generator/gru/TensorArrayV2_1TensorListReserve:query_generator/gru/TensorArrayV2_1/element_shape:output:0*query_generator/gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#query_generator/gru/TensorArrayV2_1v
query_generator/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
query_generator/gru/time?
,query_generator/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,query_generator/gru/while/maximum_iterations?
&query_generator/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&query_generator/gru/while/loop_counter?
query_generator/gru/whileWhile/query_generator/gru/while/loop_counter:output:05query_generator/gru/while/maximum_iterations:output:0!query_generator/gru/time:output:0,query_generator/gru/TensorArrayV2_1:handle:0query_generator/zeros:output:0*query_generator/gru/strided_slice:output:0Kquery_generator/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:04query_generator_gru_gru_cell_readvariableop_resource;query_generator_gru_gru_cell_matmul_readvariableop_resource=query_generator_gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*/
body'R%
#query_generator_gru_while_body_9369*/
cond'R%
#query_generator_gru_while_cond_9368*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
query_generator/gru/while?
Dquery_generator/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2F
Dquery_generator/gru/TensorArrayV2Stack/TensorListStack/element_shape?
6query_generator/gru/TensorArrayV2Stack/TensorListStackTensorListStack"query_generator/gru/while:output:3Mquery_generator/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype028
6query_generator/gru/TensorArrayV2Stack/TensorListStack?
)query_generator/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)query_generator/gru/strided_slice_2/stack?
+query_generator/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+query_generator/gru/strided_slice_2/stack_1?
+query_generator/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+query_generator/gru/strided_slice_2/stack_2?
#query_generator/gru/strided_slice_2StridedSlice?query_generator/gru/TensorArrayV2Stack/TensorListStack:tensor:02query_generator/gru/strided_slice_2/stack:output:04query_generator/gru/strided_slice_2/stack_1:output:04query_generator/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2%
#query_generator/gru/strided_slice_2?
$query_generator/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$query_generator/gru/transpose_1/perm?
query_generator/gru/transpose_1	Transpose?query_generator/gru/TensorArrayV2Stack/TensorListStack:tensor:0-query_generator/gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2!
query_generator/gru/transpose_1?
query_generator/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
query_generator/gru/runtime?
.query_generator/dense/Tensordot/ReadVariableOpReadVariableOp7query_generator_dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.query_generator/dense/Tensordot/ReadVariableOp?
$query_generator/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$query_generator/dense/Tensordot/axes?
$query_generator/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$query_generator/dense/Tensordot/free?
%query_generator/dense/Tensordot/ShapeShape#query_generator/gru/transpose_1:y:0*
T0*
_output_shapes
:2'
%query_generator/dense/Tensordot/Shape?
-query_generator/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-query_generator/dense/Tensordot/GatherV2/axis?
(query_generator/dense/Tensordot/GatherV2GatherV2.query_generator/dense/Tensordot/Shape:output:0-query_generator/dense/Tensordot/free:output:06query_generator/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(query_generator/dense/Tensordot/GatherV2?
/query_generator/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/query_generator/dense/Tensordot/GatherV2_1/axis?
*query_generator/dense/Tensordot/GatherV2_1GatherV2.query_generator/dense/Tensordot/Shape:output:0-query_generator/dense/Tensordot/axes:output:08query_generator/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*query_generator/dense/Tensordot/GatherV2_1?
%query_generator/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%query_generator/dense/Tensordot/Const?
$query_generator/dense/Tensordot/ProdProd1query_generator/dense/Tensordot/GatherV2:output:0.query_generator/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$query_generator/dense/Tensordot/Prod?
'query_generator/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'query_generator/dense/Tensordot/Const_1?
&query_generator/dense/Tensordot/Prod_1Prod3query_generator/dense/Tensordot/GatherV2_1:output:00query_generator/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&query_generator/dense/Tensordot/Prod_1?
+query_generator/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+query_generator/dense/Tensordot/concat/axis?
&query_generator/dense/Tensordot/concatConcatV2-query_generator/dense/Tensordot/free:output:0-query_generator/dense/Tensordot/axes:output:04query_generator/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&query_generator/dense/Tensordot/concat?
%query_generator/dense/Tensordot/stackPack-query_generator/dense/Tensordot/Prod:output:0/query_generator/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%query_generator/dense/Tensordot/stack?
)query_generator/dense/Tensordot/transpose	Transpose#query_generator/gru/transpose_1:y:0/query_generator/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)query_generator/dense/Tensordot/transpose?
'query_generator/dense/Tensordot/ReshapeReshape-query_generator/dense/Tensordot/transpose:y:0.query_generator/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'query_generator/dense/Tensordot/Reshape?
&query_generator/dense/Tensordot/MatMulMatMul0query_generator/dense/Tensordot/Reshape:output:06query_generator/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&query_generator/dense/Tensordot/MatMul?
'query_generator/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'query_generator/dense/Tensordot/Const_2?
-query_generator/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-query_generator/dense/Tensordot/concat_1/axis?
(query_generator/dense/Tensordot/concat_1ConcatV21query_generator/dense/Tensordot/GatherV2:output:00query_generator/dense/Tensordot/Const_2:output:06query_generator/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(query_generator/dense/Tensordot/concat_1?
query_generator/dense/TensordotReshape0query_generator/dense/Tensordot/MatMul:product:01query_generator/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
query_generator/dense/Tensordot?
,query_generator/dense/BiasAdd/ReadVariableOpReadVariableOp5query_generator_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,query_generator/dense/BiasAdd/ReadVariableOp?
query_generator/dense/BiasAddBiasAdd(query_generator/dense/Tensordot:output:04query_generator/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
query_generator/dense/BiasAdd
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSlice&query_generator/dense/BiasAdd:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/ys
truedivRealDivstrided_slice:output:0truediv/y:output:0*
T0*
_output_shapes
:	?2	
truedivQ
addAddV2truediv:z:0add_y*
T0*
_output_shapes
:	?2
add?
#categorical/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :2%
#categorical/Multinomial/num_samples?
categorical/MultinomialMultinomialadd:z:0,categorical/Multinomial/num_samples:output:0*
T0*
_output_shapes

:2
categorical/Multinomial?
SqueezeSqueeze categorical/Multinomial:output:0*
T0	*
_output_shapes
:*
squeeze_dims

?????????2	
Squeeze?
8string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleSqueeze:output:0Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0	*

Tout0*
_output_shapes
:2:
8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
IdentityIdentityAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0-^query_generator/dense/BiasAdd/ReadVariableOp/^query_generator/dense/Tensordot/ReadVariableOp+^query_generator/embedding/embedding_lookup3^query_generator/gru/gru_cell/MatMul/ReadVariableOp5^query_generator/gru/gru_cell/MatMul_1/ReadVariableOp,^query_generator/gru/gru_cell/ReadVariableOp^query_generator/gru/while7^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*
_output_shapes
:2

Identity?

Identity_1Identity"query_generator/gru/while:output:4-^query_generator/dense/BiasAdd/ReadVariableOp/^query_generator/dense/Tensordot/ReadVariableOp+^query_generator/embedding/embedding_lookup3^query_generator/gru/gru_cell/MatMul/ReadVariableOp5^query_generator/gru/gru_cell/MatMul_1/ReadVariableOp,^query_generator/gru/gru_cell/ReadVariableOp^query_generator/gru/while7^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:: : : : : : : : :?: : 2\
,query_generator/dense/BiasAdd/ReadVariableOp,query_generator/dense/BiasAdd/ReadVariableOp2`
.query_generator/dense/Tensordot/ReadVariableOp.query_generator/dense/Tensordot/ReadVariableOp2X
*query_generator/embedding/embedding_lookup*query_generator/embedding/embedding_lookup2h
2query_generator/gru/gru_cell/MatMul/ReadVariableOp2query_generator/gru/gru_cell/MatMul/ReadVariableOp2l
4query_generator/gru/gru_cell/MatMul_1/ReadVariableOp4query_generator/gru/gru_cell/MatMul_1/ReadVariableOp2Z
+query_generator/gru/gru_cell/ReadVariableOp+query_generator/gru/gru_cell/ReadVariableOp26
query_generator/gru/whilequery_generator/gru/while2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV22t
8string_lookup_1/None_lookup_table_find/LookupTableFindV28string_lookup_1/None_lookup_table_find/LookupTableFindV2:B >

_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: :!	

_output_shapes	
:?:

_output_shapes
: 
??
?
J__inference_query_generator_layer_call_and_return_conditional_losses_10127

inputs	2
embedding_embedding_lookup_9942:	?d7
$gru_gru_cell_readvariableop_resource:	?>
+gru_gru_cell_matmul_readvariableop_resource:	d?A
-gru_gru_cell_matmul_1_readvariableop_resource:
??;
'dense_tensordot_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?embedding/embedding_lookup?"gru/gru_cell/MatMul/ReadVariableOp?$gru/gru_cell/MatMul_1/ReadVariableOp?gru/gru_cell/ReadVariableOp?	gru/while?
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_9942inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/9942*+
_output_shapes
:?????????Kd*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/9942*+
_output_shapes
:?????????Kd2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????Kd2'
%embedding/embedding_lookup/Identity_1l
ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0gru/transpose/perm:output:0*
T0*+
_output_shapes
:K?????????d2
gru/transposeW
	gru/ShapeShapegru/transpose:y:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/transpose:y:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru/strided_slice_1?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru/gru_cell/ReadVariableOp?
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/gru_cell/unstack?
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru/gru_cell/MatMul/ReadVariableOp?
gru/gru_cell/MatMulMatMulgru/strided_slice_1:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd?
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/gru_cell/split/split_dim?
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/gru_cell/split?
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$gru/gru_cell/MatMul_1/ReadVariableOp?
gru/gru_cell/MatMul_1MatMulzeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_1?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_1}
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru/gru_cell/Const?
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru/gru_cell/split_1/split_dim?
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/gru_cell/split_1?
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add?
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid?
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_1?
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid_1?
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul?
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_2y
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Relu?
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_1m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/sub?
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_2?
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_3?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0zeros:output:0gru/strided_slice:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	* 
bodyR
gru_while_body_10012* 
condR
gru_while_cond_10011*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:K??????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????K?2
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freeq
dense/Tensordot/ShapeShapegru/transpose_1:y:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposegru/transpose_1:y:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????K?2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/Tensordot/MatMul}
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????K?2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????K?2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while:O K
'
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
B__inference_gru_cell_layer_call_and_return_conditional_losses_8140

inputs

states*
readvariableop_resource:	?1
matmul_readvariableop_resource:	d?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3?
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
*
__inference_<lambda>_11483
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
/__inference_query_generator_layer_call_fn_10537

inputs	
unknown:	?d
	unknown_0:	?
	unknown_1:	d?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????K?*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_query_generator_layer_call_and_return_conditional_losses_88842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????K
 
_user_specified_nameinputs
?O
?
>__inference_gru_layer_call_and_return_conditional_losses_11049

inputs
initial_state_03
 gru_cell_readvariableop_resource:	?:
'gru_cell_matmul_readvariableop_resource:	d?=
)gru_cell_matmul_1_readvariableop_resource:
??
identity

identity_1??gru_cell/MatMul/ReadVariableOp? gru_cell/MatMul_1/ReadVariableOp?gru_cell/ReadVariableOp?whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:K?????????d2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
gru_cell/MatMul/ReadVariableOp?
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split/split_dim?
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split?
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp?
gru_cell/MatMul_1MatMulinitial_state_0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1u
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell/Const?
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split_1/split_dim?
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split_1?
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Relu?
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0initial_state_0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub?
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_state_0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10959*
condR
while_cond_10958*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:K??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????K?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*,
_output_shapes
:?????????K?2

Identity?

Identity_1Identitywhile:output:4^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????Kd:??????????: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????Kd
 
_user_specified_nameinputs:YU
(
_output_shapes
:??????????
)
_user_specified_nameinitial_state/0
?

?
(__inference_gru_cell_layer_call_fn_11424

inputs
states_0
unknown:	?
	unknown_0:	d?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_82862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
%__inference_dense_layer_call_fn_11288

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????K?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_88772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????K?: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????K?
 
_user_specified_nameinputs
?
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_11357

inputs
states_0*
readvariableop_resource:	?1
matmul_readvariableop_resource:	d?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3?
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
O
__inference__creator_11308
identity: ??string_lookup_index_table?
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_2*
value_dtype0	2
string_lookup_index_table?
IdentityIdentity(string_lookup_index_table:table_handle:0^string_lookup_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 26
string_lookup_index_tablestring_lookup_index_table
?
?
__inference_save_fn_11443
checkpoint_key[
Wstring_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2	

identity_3

identity_4

identity_5??Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0	*
Tvalues0*
_output_shapes

::2L
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
while_cond_8747
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_8747___redundant_placeholder02
.while_while_cond_8747___redundant_placeholder12
.while_while_cond_8747___redundant_placeholder22
.while_while_cond_8747___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
.
__inference__initializer_11298
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
while_cond_11104
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_11104___redundant_placeholder03
/while_while_cond_11104___redundant_placeholder13
/while_while_cond_11104___redundant_placeholder23
/while_while_cond_11104___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
*
__inference_<lambda>_11488
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?

"__inference_generate_one_step_9745

inputs

statesG
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	B
/query_generator_embedding_embedding_lookup_9558:	?dG
4query_generator_gru_gru_cell_readvariableop_resource:	?N
;query_generator_gru_gru_cell_matmul_readvariableop_resource:	d?Q
=query_generator_gru_gru_cell_matmul_1_readvariableop_resource:
??K
7query_generator_dense_tensordot_readvariableop_resource:
??D
5query_generator_dense_biasadd_readvariableop_resource:	?	
add_yI
Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value
identity

identity_1??,query_generator/dense/BiasAdd/ReadVariableOp?.query_generator/dense/Tensordot/ReadVariableOp?*query_generator/embedding/embedding_lookup?2query_generator/gru/gru_cell/MatMul/ReadVariableOp?4query_generator/gru/gru_cell/MatMul_1/ReadVariableOp?+query_generator/gru/gru_cell/ReadVariableOp?query_generator/gru/while?6string_lookup/None_lookup_table_find/LookupTableFindV2?8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
UnicodeSplit/Reshape/shape?
UnicodeSplit/ReshapeReshapeinputs#UnicodeSplit/Reshape/shape:output:0*
T0*
_output_shapes
:2
UnicodeSplit/Reshape?
UnicodeSplit/UnicodeDecodeUnicodeDecodeUnicodeSplit/Reshape:output:0*)
_output_shapes
::?????????*
input_encodingUTF-82
UnicodeSplit/UnicodeDecode?
,UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,UnicodeSplit/RaggedExpandDims/ExpandDims/dim?
(UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims(UnicodeSplit/UnicodeDecode:char_values:05UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2*
(UnicodeSplit/RaggedExpandDims/ExpandDims?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2A
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape?
MUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
MUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2?
GUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0VUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2I
GUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2?
=UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMulRUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: 2?
=UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_mask2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackAUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0?
EUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
EUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis?
@UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0NUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:2B
@UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat?
AUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:?????????2C
AUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2K
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R2A
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const?
ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShapeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	2\
ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape?
hUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2j
hUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack?
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2l
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1?
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2l
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2?
bUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicecUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0qUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2d
bUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice?
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2}
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y?
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: 2{
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/CastCast?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1Cast?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1?
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRange?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast:y:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2}
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range?
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMul?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0HUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:?????????2{
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul?
DUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncodeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:?????????*
output_encodingUTF-82F
DUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleMUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2`
string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2!
string_lookup/assert_equal/NoOp?
string_lookup/IdentityIdentity?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
string_lookup/Identity_1Identity'UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
_output_shapes
:2
string_lookup/Identity_1n
RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
RaggedToTensor/zerosw
RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
RaggedToTensor/Const?
#RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorRaggedToTensor/Const:output:0string_lookup/Identity:output:0RaggedToTensor/zeros:output:0!string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2%
#RaggedToTensor/RaggedTensorToTensor?
*query_generator/embedding/embedding_lookupResourceGather/query_generator_embedding_embedding_lookup_9558,RaggedToTensor/RaggedTensorToTensor:result:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*B
_class8
64loc:@query_generator/embedding/embedding_lookup/9558*+
_output_shapes
:?????????d*
dtype02,
*query_generator/embedding/embedding_lookup?
3query_generator/embedding/embedding_lookup/IdentityIdentity3query_generator/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@query_generator/embedding/embedding_lookup/9558*+
_output_shapes
:?????????d25
3query_generator/embedding/embedding_lookup/Identity?
5query_generator/embedding/embedding_lookup/Identity_1Identity<query_generator/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d27
5query_generator/embedding/embedding_lookup/Identity_1?
"query_generator/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"query_generator/gru/transpose/perm?
query_generator/gru/transpose	Transpose>query_generator/embedding/embedding_lookup/Identity_1:output:0+query_generator/gru/transpose/perm:output:0*
T0*+
_output_shapes
:?????????d2
query_generator/gru/transpose?
query_generator/gru/ShapeShape!query_generator/gru/transpose:y:0*
T0*
_output_shapes
:2
query_generator/gru/Shape?
'query_generator/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'query_generator/gru/strided_slice/stack?
)query_generator/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)query_generator/gru/strided_slice/stack_1?
)query_generator/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)query_generator/gru/strided_slice/stack_2?
!query_generator/gru/strided_sliceStridedSlice"query_generator/gru/Shape:output:00query_generator/gru/strided_slice/stack:output:02query_generator/gru/strided_slice/stack_1:output:02query_generator/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!query_generator/gru/strided_slice?
/query_generator/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/query_generator/gru/TensorArrayV2/element_shape?
!query_generator/gru/TensorArrayV2TensorListReserve8query_generator/gru/TensorArrayV2/element_shape:output:0*query_generator/gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!query_generator/gru/TensorArrayV2?
Iquery_generator/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   d   2K
Iquery_generator/gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
;query_generator/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!query_generator/gru/transpose:y:0Rquery_generator/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;query_generator/gru/TensorArrayUnstack/TensorListFromTensor?
)query_generator/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)query_generator/gru/strided_slice_1/stack?
+query_generator/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+query_generator/gru/strided_slice_1/stack_1?
+query_generator/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+query_generator/gru/strided_slice_1/stack_2?
#query_generator/gru/strided_slice_1StridedSlice!query_generator/gru/transpose:y:02query_generator/gru/strided_slice_1/stack:output:04query_generator/gru/strided_slice_1/stack_1:output:04query_generator/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_mask2%
#query_generator/gru/strided_slice_1?
+query_generator/gru/gru_cell/ReadVariableOpReadVariableOp4query_generator_gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+query_generator/gru/gru_cell/ReadVariableOp?
$query_generator/gru/gru_cell/unstackUnpack3query_generator/gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2&
$query_generator/gru/gru_cell/unstack?
2query_generator/gru/gru_cell/MatMul/ReadVariableOpReadVariableOp;query_generator_gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype024
2query_generator/gru/gru_cell/MatMul/ReadVariableOp?
#query_generator/gru/gru_cell/MatMulMatMul,query_generator/gru/strided_slice_1:output:0:query_generator/gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2%
#query_generator/gru/gru_cell/MatMul?
$query_generator/gru/gru_cell/BiasAddBiasAdd-query_generator/gru/gru_cell/MatMul:product:0-query_generator/gru/gru_cell/unstack:output:0*
T0*
_output_shapes
:	?2&
$query_generator/gru/gru_cell/BiasAdd?
,query_generator/gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,query_generator/gru/gru_cell/split/split_dim?
"query_generator/gru/gru_cell/splitSplit5query_generator/gru/gru_cell/split/split_dim:output:0-query_generator/gru/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_split2$
"query_generator/gru/gru_cell/split?
4query_generator/gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp=query_generator_gru_gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4query_generator/gru/gru_cell/MatMul_1/ReadVariableOp?
%query_generator/gru/gru_cell/MatMul_1MatMulstates<query_generator/gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%query_generator/gru/gru_cell/MatMul_1?
&query_generator/gru/gru_cell/BiasAdd_1BiasAdd/query_generator/gru/gru_cell/MatMul_1:product:0-query_generator/gru/gru_cell/unstack:output:1*
T0*
_output_shapes
:	?2(
&query_generator/gru/gru_cell/BiasAdd_1?
"query_generator/gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2$
"query_generator/gru/gru_cell/Const?
.query_generator/gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.query_generator/gru/gru_cell/split_1/split_dim?
$query_generator/gru/gru_cell/split_1SplitV/query_generator/gru/gru_cell/BiasAdd_1:output:0+query_generator/gru/gru_cell/Const:output:07query_generator/gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	?:	?:	?*
	num_split2&
$query_generator/gru/gru_cell/split_1?
 query_generator/gru/gru_cell/addAddV2+query_generator/gru/gru_cell/split:output:0-query_generator/gru/gru_cell/split_1:output:0*
T0*
_output_shapes
:	?2"
 query_generator/gru/gru_cell/add?
$query_generator/gru/gru_cell/SigmoidSigmoid$query_generator/gru/gru_cell/add:z:0*
T0*
_output_shapes
:	?2&
$query_generator/gru/gru_cell/Sigmoid?
"query_generator/gru/gru_cell/add_1AddV2+query_generator/gru/gru_cell/split:output:1-query_generator/gru/gru_cell/split_1:output:1*
T0*
_output_shapes
:	?2$
"query_generator/gru/gru_cell/add_1?
&query_generator/gru/gru_cell/Sigmoid_1Sigmoid&query_generator/gru/gru_cell/add_1:z:0*
T0*
_output_shapes
:	?2(
&query_generator/gru/gru_cell/Sigmoid_1?
 query_generator/gru/gru_cell/mulMul*query_generator/gru/gru_cell/Sigmoid_1:y:0-query_generator/gru/gru_cell/split_1:output:2*
T0*
_output_shapes
:	?2"
 query_generator/gru/gru_cell/mul?
"query_generator/gru/gru_cell/add_2AddV2+query_generator/gru/gru_cell/split:output:2$query_generator/gru/gru_cell/mul:z:0*
T0*
_output_shapes
:	?2$
"query_generator/gru/gru_cell/add_2?
!query_generator/gru/gru_cell/ReluRelu&query_generator/gru/gru_cell/add_2:z:0*
T0*
_output_shapes
:	?2#
!query_generator/gru/gru_cell/Relu?
"query_generator/gru/gru_cell/mul_1Mul(query_generator/gru/gru_cell/Sigmoid:y:0states*
T0*
_output_shapes
:	?2$
"query_generator/gru/gru_cell/mul_1?
"query_generator/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"query_generator/gru/gru_cell/sub/x?
 query_generator/gru/gru_cell/subSub+query_generator/gru/gru_cell/sub/x:output:0(query_generator/gru/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	?2"
 query_generator/gru/gru_cell/sub?
"query_generator/gru/gru_cell/mul_2Mul$query_generator/gru/gru_cell/sub:z:0/query_generator/gru/gru_cell/Relu:activations:0*
T0*
_output_shapes
:	?2$
"query_generator/gru/gru_cell/mul_2?
"query_generator/gru/gru_cell/add_3AddV2&query_generator/gru/gru_cell/mul_1:z:0&query_generator/gru/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	?2$
"query_generator/gru/gru_cell/add_3?
1query_generator/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   23
1query_generator/gru/TensorArrayV2_1/element_shape?
#query_generator/gru/TensorArrayV2_1TensorListReserve:query_generator/gru/TensorArrayV2_1/element_shape:output:0*query_generator/gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#query_generator/gru/TensorArrayV2_1v
query_generator/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
query_generator/gru/time?
,query_generator/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,query_generator/gru/while/maximum_iterations?
&query_generator/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&query_generator/gru/while/loop_counter?
query_generator/gru/whileWhile/query_generator/gru/while/loop_counter:output:05query_generator/gru/while/maximum_iterations:output:0!query_generator/gru/time:output:0,query_generator/gru/TensorArrayV2_1:handle:0states*query_generator/gru/strided_slice:output:0Kquery_generator/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:04query_generator_gru_gru_cell_readvariableop_resource;query_generator_gru_gru_cell_matmul_readvariableop_resource=query_generator_gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*/
body'R%
#query_generator_gru_while_body_9615*/
cond'R%
#query_generator_gru_while_cond_9614*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
query_generator/gru/while?
Dquery_generator/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2F
Dquery_generator/gru/TensorArrayV2Stack/TensorListStack/element_shape?
6query_generator/gru/TensorArrayV2Stack/TensorListStackTensorListStack"query_generator/gru/while:output:3Mquery_generator/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype028
6query_generator/gru/TensorArrayV2Stack/TensorListStack?
)query_generator/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)query_generator/gru/strided_slice_2/stack?
+query_generator/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+query_generator/gru/strided_slice_2/stack_1?
+query_generator/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+query_generator/gru/strided_slice_2/stack_2?
#query_generator/gru/strided_slice_2StridedSlice?query_generator/gru/TensorArrayV2Stack/TensorListStack:tensor:02query_generator/gru/strided_slice_2/stack:output:04query_generator/gru/strided_slice_2/stack_1:output:04query_generator/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2%
#query_generator/gru/strided_slice_2?
$query_generator/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$query_generator/gru/transpose_1/perm?
query_generator/gru/transpose_1	Transpose?query_generator/gru/TensorArrayV2Stack/TensorListStack:tensor:0-query_generator/gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2!
query_generator/gru/transpose_1?
query_generator/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
query_generator/gru/runtime?
.query_generator/dense/Tensordot/ReadVariableOpReadVariableOp7query_generator_dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.query_generator/dense/Tensordot/ReadVariableOp?
$query_generator/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$query_generator/dense/Tensordot/axes?
$query_generator/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$query_generator/dense/Tensordot/free?
%query_generator/dense/Tensordot/ShapeShape#query_generator/gru/transpose_1:y:0*
T0*
_output_shapes
:2'
%query_generator/dense/Tensordot/Shape?
-query_generator/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-query_generator/dense/Tensordot/GatherV2/axis?
(query_generator/dense/Tensordot/GatherV2GatherV2.query_generator/dense/Tensordot/Shape:output:0-query_generator/dense/Tensordot/free:output:06query_generator/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(query_generator/dense/Tensordot/GatherV2?
/query_generator/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/query_generator/dense/Tensordot/GatherV2_1/axis?
*query_generator/dense/Tensordot/GatherV2_1GatherV2.query_generator/dense/Tensordot/Shape:output:0-query_generator/dense/Tensordot/axes:output:08query_generator/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*query_generator/dense/Tensordot/GatherV2_1?
%query_generator/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%query_generator/dense/Tensordot/Const?
$query_generator/dense/Tensordot/ProdProd1query_generator/dense/Tensordot/GatherV2:output:0.query_generator/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$query_generator/dense/Tensordot/Prod?
'query_generator/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'query_generator/dense/Tensordot/Const_1?
&query_generator/dense/Tensordot/Prod_1Prod3query_generator/dense/Tensordot/GatherV2_1:output:00query_generator/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&query_generator/dense/Tensordot/Prod_1?
+query_generator/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+query_generator/dense/Tensordot/concat/axis?
&query_generator/dense/Tensordot/concatConcatV2-query_generator/dense/Tensordot/free:output:0-query_generator/dense/Tensordot/axes:output:04query_generator/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&query_generator/dense/Tensordot/concat?
%query_generator/dense/Tensordot/stackPack-query_generator/dense/Tensordot/Prod:output:0/query_generator/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%query_generator/dense/Tensordot/stack?
)query_generator/dense/Tensordot/transpose	Transpose#query_generator/gru/transpose_1:y:0/query_generator/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)query_generator/dense/Tensordot/transpose?
'query_generator/dense/Tensordot/ReshapeReshape-query_generator/dense/Tensordot/transpose:y:0.query_generator/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'query_generator/dense/Tensordot/Reshape?
&query_generator/dense/Tensordot/MatMulMatMul0query_generator/dense/Tensordot/Reshape:output:06query_generator/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&query_generator/dense/Tensordot/MatMul?
'query_generator/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'query_generator/dense/Tensordot/Const_2?
-query_generator/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-query_generator/dense/Tensordot/concat_1/axis?
(query_generator/dense/Tensordot/concat_1ConcatV21query_generator/dense/Tensordot/GatherV2:output:00query_generator/dense/Tensordot/Const_2:output:06query_generator/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(query_generator/dense/Tensordot/concat_1?
query_generator/dense/TensordotReshape0query_generator/dense/Tensordot/MatMul:product:01query_generator/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
query_generator/dense/Tensordot?
,query_generator/dense/BiasAdd/ReadVariableOpReadVariableOp5query_generator_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,query_generator/dense/BiasAdd/ReadVariableOp?
query_generator/dense/BiasAddBiasAdd(query_generator/dense/Tensordot:output:04query_generator/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
query_generator/dense/BiasAdd
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSlice&query_generator/dense/BiasAdd:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/ys
truedivRealDivstrided_slice:output:0truediv/y:output:0*
T0*
_output_shapes
:	?2	
truedivQ
addAddV2truediv:z:0add_y*
T0*
_output_shapes
:	?2
add?
#categorical/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :2%
#categorical/Multinomial/num_samples?
categorical/MultinomialMultinomialadd:z:0,categorical/Multinomial/num_samples:output:0*
T0*
_output_shapes

:2
categorical/Multinomial?
SqueezeSqueeze categorical/Multinomial:output:0*
T0	*
_output_shapes
:*
squeeze_dims

?????????2	
Squeeze?
8string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleSqueeze:output:0Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0	*

Tout0*
_output_shapes
:2:
8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
IdentityIdentityAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0-^query_generator/dense/BiasAdd/ReadVariableOp/^query_generator/dense/Tensordot/ReadVariableOp+^query_generator/embedding/embedding_lookup3^query_generator/gru/gru_cell/MatMul/ReadVariableOp5^query_generator/gru/gru_cell/MatMul_1/ReadVariableOp,^query_generator/gru/gru_cell/ReadVariableOp^query_generator/gru/while7^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*
_output_shapes
:2

Identity?

Identity_1Identity"query_generator/gru/while:output:4-^query_generator/dense/BiasAdd/ReadVariableOp/^query_generator/dense/Tensordot/ReadVariableOp+^query_generator/embedding/embedding_lookup3^query_generator/gru/gru_cell/MatMul/ReadVariableOp5^query_generator/gru/gru_cell/MatMul_1/ReadVariableOp,^query_generator/gru/gru_cell/ReadVariableOp^query_generator/gru/while7^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*
_output_shapes
:	?2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,::	?: : : : : : : : :?: : 2\
,query_generator/dense/BiasAdd/ReadVariableOp,query_generator/dense/BiasAdd/ReadVariableOp2`
.query_generator/dense/Tensordot/ReadVariableOp.query_generator/dense/Tensordot/ReadVariableOp2X
*query_generator/embedding/embedding_lookup*query_generator/embedding/embedding_lookup2h
2query_generator/gru/gru_cell/MatMul/ReadVariableOp2query_generator/gru/gru_cell/MatMul/ReadVariableOp2l
4query_generator/gru/gru_cell/MatMul_1/ReadVariableOp4query_generator/gru/gru_cell/MatMul_1/ReadVariableOp2Z
+query_generator/gru/gru_cell/ReadVariableOp+query_generator/gru/gru_cell/ReadVariableOp26
query_generator/gru/whilequery_generator/gru/while2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV22t
8string_lookup_1/None_lookup_table_find/LookupTableFindV28string_lookup_1/None_lookup_table_find/LookupTableFindV2:B >

_output_shapes
:
 
_user_specified_nameinputs:GC

_output_shapes
:	?
 
_user_specified_namestates:

_output_shapes
: :!


_output_shapes	
:?:

_output_shapes
: 
?
?
while_cond_8981
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_8981___redundant_placeholder02
.while_while_cond_8981___redundant_placeholder12
.while_while_cond_8981___redundant_placeholder22
.while_while_cond_8981___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
/__inference_query_generator_layer_call_fn_10520
input_1	
unknown:	?d
	unknown_0:	?
	unknown_1:	d?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????K?*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_query_generator_layer_call_and_return_conditional_losses_88842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????K
!
_user_specified_name	input_1
?	
?
#__inference_gru_layer_call_fn_11221
inputs_0
unknown:	?
	unknown_0:	d?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:???????????????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_84202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????d
"
_user_specified_name
inputs/0
?
?
while_cond_10654
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10654___redundant_placeholder03
/while_while_cond_10654___redundant_placeholder13
/while_while_cond_10654___redundant_placeholder23
/while_while_cond_10654___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?!
?
while_body_8355
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_gru_cell_8377_0:	?(
while_gru_cell_8379_0:	d?)
while_gru_cell_8381_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_gru_cell_8377:	?&
while_gru_cell_8379:	d?'
while_gru_cell_8381:
????&while/gru_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_8377_0while_gru_cell_8379_0while_gru_cell_8381_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_82862(
&while/gru_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4",
while_gru_cell_8377while_gru_cell_8377_0",
while_gru_cell_8379while_gru_cell_8379_0",
while_gru_cell_8381while_gru_cell_8381_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?H
?
__inference__traced_save_11599
file_prefixU
Qsavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2	W
Ssavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_1S
Osavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2U
Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1	C
?savev2_query_generator_embedding_embeddings_read_readvariableop;
7savev2_query_generator_dense_kernel_read_readvariableop9
5savev2_query_generator_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopB
>savev2_query_generator_gru_gru_cell_kernel_read_readvariableopL
Hsavev2_query_generator_gru_gru_cell_recurrent_kernel_read_readvariableop@
<savev2_query_generator_gru_gru_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopJ
Fsavev2_adam_query_generator_embedding_embeddings_m_read_readvariableopB
>savev2_adam_query_generator_dense_kernel_m_read_readvariableop@
<savev2_adam_query_generator_dense_bias_m_read_readvariableopI
Esavev2_adam_query_generator_gru_gru_cell_kernel_m_read_readvariableopS
Osavev2_adam_query_generator_gru_gru_cell_recurrent_kernel_m_read_readvariableopG
Csavev2_adam_query_generator_gru_gru_cell_bias_m_read_readvariableopJ
Fsavev2_adam_query_generator_embedding_embeddings_v_read_readvariableopB
>savev2_adam_query_generator_dense_kernel_v_read_readvariableop@
<savev2_adam_query_generator_dense_bias_v_read_readvariableopI
Esavev2_adam_query_generator_gru_gru_cell_kernel_v_read_readvariableopS
Osavev2_adam_query_generator_gru_gru_cell_recurrent_kernel_v_read_readvariableopG
Csavev2_adam_query_generator_gru_gru_cell_bias_v_read_readvariableop
savev2_const_3

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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B,chars_from_ids/_table/.ATTRIBUTES/table-keysB.chars_from_ids/_table/.ATTRIBUTES/table-valuesB,ids_from_chars/_table/.ATTRIBUTES/table-keysB.ids_from_chars/_table/.ATTRIBUTES/table-valuesB5model/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB-model/dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB+model/dense/bias/.ATTRIBUTES/VARIABLE_VALUEB/model/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB1model/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB1model/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB0model/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/3/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Qsavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_1Osavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1?savev2_query_generator_embedding_embeddings_read_readvariableop7savev2_query_generator_dense_kernel_read_readvariableop5savev2_query_generator_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop>savev2_query_generator_gru_gru_cell_kernel_read_readvariableopHsavev2_query_generator_gru_gru_cell_recurrent_kernel_read_readvariableop<savev2_query_generator_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopFsavev2_adam_query_generator_embedding_embeddings_m_read_readvariableop>savev2_adam_query_generator_dense_kernel_m_read_readvariableop<savev2_adam_query_generator_dense_bias_m_read_readvariableopEsavev2_adam_query_generator_gru_gru_cell_kernel_m_read_readvariableopOsavev2_adam_query_generator_gru_gru_cell_recurrent_kernel_m_read_readvariableopCsavev2_adam_query_generator_gru_gru_cell_bias_m_read_readvariableopFsavev2_adam_query_generator_embedding_embeddings_v_read_readvariableop>savev2_adam_query_generator_dense_kernel_v_read_readvariableop<savev2_adam_query_generator_dense_bias_v_read_readvariableopEsavev2_adam_query_generator_gru_gru_cell_kernel_v_read_readvariableopOsavev2_adam_query_generator_gru_gru_cell_recurrent_kernel_v_read_readvariableopCsavev2_adam_query_generator_gru_gru_cell_bias_v_read_readvariableopsavev2_const_3"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2			2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::	?d:
??:?: : : : : :	d?:
??:	?: : :	?d:
??:?:	d?:
??:	?:	?d:
??:?:	d?:
??:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::%!

_output_shapes
:	?d:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	d?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?d:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	d?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:%!

_output_shapes
:	?d:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	d?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:

_output_shapes
: 
?
.
__inference__initializer_11313
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?d
?
#query_generator_gru_while_body_9369D
@query_generator_gru_while_query_generator_gru_while_loop_counterJ
Fquery_generator_gru_while_query_generator_gru_while_maximum_iterations)
%query_generator_gru_while_placeholder+
'query_generator_gru_while_placeholder_1+
'query_generator_gru_while_placeholder_2A
=query_generator_gru_while_query_generator_gru_strided_slice_0
{query_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensor_0O
<query_generator_gru_while_gru_cell_readvariableop_resource_0:	?V
Cquery_generator_gru_while_gru_cell_matmul_readvariableop_resource_0:	d?Y
Equery_generator_gru_while_gru_cell_matmul_1_readvariableop_resource_0:
??&
"query_generator_gru_while_identity(
$query_generator_gru_while_identity_1(
$query_generator_gru_while_identity_2(
$query_generator_gru_while_identity_3(
$query_generator_gru_while_identity_4?
;query_generator_gru_while_query_generator_gru_strided_slice}
yquery_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensorM
:query_generator_gru_while_gru_cell_readvariableop_resource:	?T
Aquery_generator_gru_while_gru_cell_matmul_readvariableop_resource:	d?W
Cquery_generator_gru_while_gru_cell_matmul_1_readvariableop_resource:
????8query_generator/gru/while/gru_cell/MatMul/ReadVariableOp?:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp?1query_generator/gru/while/gru_cell/ReadVariableOp?
Kquery_generator/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   d   2M
Kquery_generator/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=query_generator/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{query_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensor_0%query_generator_gru_while_placeholderTquery_generator/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:d*
element_dtype02?
=query_generator/gru/while/TensorArrayV2Read/TensorListGetItem?
1query_generator/gru/while/gru_cell/ReadVariableOpReadVariableOp<query_generator_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype023
1query_generator/gru/while/gru_cell/ReadVariableOp?
*query_generator/gru/while/gru_cell/unstackUnpack9query_generator/gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2,
*query_generator/gru/while/gru_cell/unstack?
8query_generator/gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOpCquery_generator_gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02:
8query_generator/gru/while/gru_cell/MatMul/ReadVariableOp?
)query_generator/gru/while/gru_cell/MatMulMatMulDquery_generator/gru/while/TensorArrayV2Read/TensorListGetItem:item:0@query_generator/gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2+
)query_generator/gru/while/gru_cell/MatMul?
*query_generator/gru/while/gru_cell/BiasAddBiasAdd3query_generator/gru/while/gru_cell/MatMul:product:03query_generator/gru/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	?2,
*query_generator/gru/while/gru_cell/BiasAdd?
2query_generator/gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2query_generator/gru/while/gru_cell/split/split_dim?
(query_generator/gru/while/gru_cell/splitSplit;query_generator/gru/while/gru_cell/split/split_dim:output:03query_generator/gru/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_split2*
(query_generator/gru/while/gru_cell/split?
:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpEquery_generator_gru_while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02<
:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp?
+query_generator/gru/while/gru_cell/MatMul_1MatMul'query_generator_gru_while_placeholder_2Bquery_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+query_generator/gru/while/gru_cell/MatMul_1?
,query_generator/gru/while/gru_cell/BiasAdd_1BiasAdd5query_generator/gru/while/gru_cell/MatMul_1:product:03query_generator/gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2.
,query_generator/gru/while/gru_cell/BiasAdd_1?
(query_generator/gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2*
(query_generator/gru/while/gru_cell/Const?
4query_generator/gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4query_generator/gru/while/gru_cell/split_1/split_dim?
*query_generator/gru/while/gru_cell/split_1SplitV5query_generator/gru/while/gru_cell/BiasAdd_1:output:01query_generator/gru/while/gru_cell/Const:output:0=query_generator/gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2,
*query_generator/gru/while/gru_cell/split_1?
&query_generator/gru/while/gru_cell/addAddV21query_generator/gru/while/gru_cell/split:output:03query_generator/gru/while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2(
&query_generator/gru/while/gru_cell/add?
*query_generator/gru/while/gru_cell/SigmoidSigmoid*query_generator/gru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2,
*query_generator/gru/while/gru_cell/Sigmoid?
(query_generator/gru/while/gru_cell/add_1AddV21query_generator/gru/while/gru_cell/split:output:13query_generator/gru/while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2*
(query_generator/gru/while/gru_cell/add_1?
,query_generator/gru/while/gru_cell/Sigmoid_1Sigmoid,query_generator/gru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2.
,query_generator/gru/while/gru_cell/Sigmoid_1?
&query_generator/gru/while/gru_cell/mulMul0query_generator/gru/while/gru_cell/Sigmoid_1:y:03query_generator/gru/while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2(
&query_generator/gru/while/gru_cell/mul?
(query_generator/gru/while/gru_cell/add_2AddV21query_generator/gru/while/gru_cell/split:output:2*query_generator/gru/while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2*
(query_generator/gru/while/gru_cell/add_2?
'query_generator/gru/while/gru_cell/ReluRelu,query_generator/gru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2)
'query_generator/gru/while/gru_cell/Relu?
(query_generator/gru/while/gru_cell/mul_1Mul.query_generator/gru/while/gru_cell/Sigmoid:y:0'query_generator_gru_while_placeholder_2*
T0*(
_output_shapes
:??????????2*
(query_generator/gru/while/gru_cell/mul_1?
(query_generator/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(query_generator/gru/while/gru_cell/sub/x?
&query_generator/gru/while/gru_cell/subSub1query_generator/gru/while/gru_cell/sub/x:output:0.query_generator/gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2(
&query_generator/gru/while/gru_cell/sub?
(query_generator/gru/while/gru_cell/mul_2Mul*query_generator/gru/while/gru_cell/sub:z:05query_generator/gru/while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2*
(query_generator/gru/while/gru_cell/mul_2?
(query_generator/gru/while/gru_cell/add_3AddV2,query_generator/gru/while/gru_cell/mul_1:z:0,query_generator/gru/while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2*
(query_generator/gru/while/gru_cell/add_3?
>query_generator/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'query_generator_gru_while_placeholder_1%query_generator_gru_while_placeholder,query_generator/gru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02@
>query_generator/gru/while/TensorArrayV2Write/TensorListSetItem?
query_generator/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
query_generator/gru/while/add/y?
query_generator/gru/while/addAddV2%query_generator_gru_while_placeholder(query_generator/gru/while/add/y:output:0*
T0*
_output_shapes
: 2
query_generator/gru/while/add?
!query_generator/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!query_generator/gru/while/add_1/y?
query_generator/gru/while/add_1AddV2@query_generator_gru_while_query_generator_gru_while_loop_counter*query_generator/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
query_generator/gru/while/add_1?
"query_generator/gru/while/IdentityIdentity#query_generator/gru/while/add_1:z:09^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2$
"query_generator/gru/while/Identity?
$query_generator/gru/while/Identity_1IdentityFquery_generator_gru_while_query_generator_gru_while_maximum_iterations9^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2&
$query_generator/gru/while/Identity_1?
$query_generator/gru/while/Identity_2Identity!query_generator/gru/while/add:z:09^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2&
$query_generator/gru/while/Identity_2?
$query_generator/gru/while/Identity_3IdentityNquery_generator/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:09^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2&
$query_generator/gru/while/Identity_3?
$query_generator/gru/while/Identity_4Identity,query_generator/gru/while/gru_cell/add_3:z:09^query_generator/gru/while/gru_cell/MatMul/ReadVariableOp;^query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2^query_generator/gru/while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2&
$query_generator/gru/while/Identity_4"?
Cquery_generator_gru_while_gru_cell_matmul_1_readvariableop_resourceEquery_generator_gru_while_gru_cell_matmul_1_readvariableop_resource_0"?
Aquery_generator_gru_while_gru_cell_matmul_readvariableop_resourceCquery_generator_gru_while_gru_cell_matmul_readvariableop_resource_0"z
:query_generator_gru_while_gru_cell_readvariableop_resource<query_generator_gru_while_gru_cell_readvariableop_resource_0"Q
"query_generator_gru_while_identity+query_generator/gru/while/Identity:output:0"U
$query_generator_gru_while_identity_1-query_generator/gru/while/Identity_1:output:0"U
$query_generator_gru_while_identity_2-query_generator/gru/while/Identity_2:output:0"U
$query_generator_gru_while_identity_3-query_generator/gru/while/Identity_3:output:0"U
$query_generator_gru_while_identity_4-query_generator/gru/while/Identity_4:output:0"|
;query_generator_gru_while_query_generator_gru_strided_slice=query_generator_gru_while_query_generator_gru_strided_slice_0"?
yquery_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensor{query_generator_gru_while_tensorarrayv2read_tensorlistgetitem_query_generator_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2t
8query_generator/gru/while/gru_cell/MatMul/ReadVariableOp8query_generator/gru/while/gru_cell/MatMul/ReadVariableOp2x
:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp:query_generator/gru/while/gru_cell/MatMul_1/ReadVariableOp2f
1query_generator/gru/while/gru_cell/ReadVariableOp1query_generator/gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_query_generator_layer_call_and_return_conditional_losses_8884

inputs	!
embedding_8677:	?d
gru_8839:	?
gru_8841:	d?
gru_8843:
??

dense_8878:
??

dense_8880:	?
identity??dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?gru/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_8677*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????Kd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_86762#
!embedding/StatefulPartitionedCallh
ShapeShape*embedding/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zeros?
gru/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0zeros:output:0gru_8839gru_8841gru_8843*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????K?:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_88382
gru/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0
dense_8878
dense_8880*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????K?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_88772
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:O K
'
_output_shapes
:?????????K
 
_user_specified_nameinputs
?E
?
while_body_10959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?B
/while_gru_cell_matmul_readvariableop_resource_0:	d?E
1while_gru_cell_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?@
-while_gru_cell_matmul_readvariableop_resource:	d?C
/while_gru_cell_matmul_1_readvariableop_resource:
????$while/gru_cell/MatMul/ReadVariableOp?&while/gru_cell/MatMul_1/ReadVariableOp?while/gru_cell/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02&
$while/gru_cell/MatMul/ReadVariableOp?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
while/gru_cell/split/split_dim?
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split?
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp?
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell/Const?
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell/split_1/split_dim?
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split_1?
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Relu?
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
gru_while_cond_9823$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2$
 gru_while_less_gru_strided_slice:
6gru_while_gru_while_cond_9823___redundant_placeholder0:
6gru_while_gru_while_cond_9823___redundant_placeholder1:
6gru_while_gru_while_cond_9823___redundant_placeholder2:
6gru_while_gru_while_cond_9823___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder gru_while_less_gru_strided_slice*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
~
)__inference_embedding_layer_call_fn_10587

inputs	
unknown:	?d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????Kd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_86762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????Kd2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????K: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????K
 
_user_specified_nameinputs
?>
?
=__inference_gru_layer_call_and_return_conditional_losses_8218

inputs 
gru_cell_8141:	? 
gru_cell_8143:	d?!
gru_cell_8145:
??
identity

identity_1?? gru_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_2?
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_8141gru_cell_8143gru_cell_8145*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_81402"
 gru_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_8141gru_cell_8143gru_cell_8145*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_8153*
condR
while_cond_8152*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0!^gru_cell/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identitywhile:output:4!^gru_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????d: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs
? 
?
@__inference_dense_layer_call_and_return_conditional_losses_11279

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????K?2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????K?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????K?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????K?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????K?
 
_user_specified_nameinputs
?E
?
while_body_10813
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?B
/while_gru_cell_matmul_readvariableop_resource_0:	d?E
1while_gru_cell_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?@
-while_gru_cell_matmul_readvariableop_resource:	d?C
/while_gru_cell_matmul_1_readvariableop_resource:
????$while/gru_cell/MatMul/ReadVariableOp?&while/gru_cell/MatMul_1/ReadVariableOp?while/gru_cell/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02&
$while/gru_cell/MatMul/ReadVariableOp?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
while/gru_cell/split/split_dim?
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split?
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp?
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell/Const?
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell/split_1/split_dim?
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split_1?
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Relu?
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?[
?
>__inference_gru_layer_call_and_return_conditional_losses_10903
inputs_03
 gru_cell_readvariableop_resource:	?:
'gru_cell_matmul_readvariableop_resource:	d?=
)gru_cell_matmul_1_readvariableop_resource:
??
identity

identity_1??gru_cell/MatMul/ReadVariableOp? gru_cell/MatMul_1/ReadVariableOp?gru_cell/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_2?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
gru_cell/MatMul/ReadVariableOp?
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split/split_dim?
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split?
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp?
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1u
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell/Const?
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split_1/split_dim?
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split_1?
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Relu?
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub?
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10813*
condR
while_cond_10812*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identitywhile:output:4^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????d: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????d
"
_user_specified_name
inputs/0
?
?
while_cond_10812
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10812___redundant_placeholder03
/while_while_cond_10812___redundant_placeholder13
/while_while_cond_10812___redundant_placeholder23
/while_while_cond_10812___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
J__inference_query_generator_layer_call_and_return_conditional_losses_10503
input_1	3
 embedding_embedding_lookup_10318:	?d7
$gru_gru_cell_readvariableop_resource:	?>
+gru_gru_cell_matmul_readvariableop_resource:	d?A
-gru_gru_cell_matmul_1_readvariableop_resource:
??;
'dense_tensordot_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?embedding/embedding_lookup?"gru/gru_cell/MatMul/ReadVariableOp?$gru/gru_cell/MatMul_1/ReadVariableOp?gru/gru_cell/ReadVariableOp?	gru/while?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_10318input_1",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/10318*+
_output_shapes
:?????????Kd*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/10318*+
_output_shapes
:?????????Kd2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????Kd2'
%embedding/embedding_lookup/Identity_1l
ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0gru/transpose/perm:output:0*
T0*+
_output_shapes
:K?????????d2
gru/transposeW
	gru/ShapeShapegru/transpose:y:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/transpose:y:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru/strided_slice_1?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru/gru_cell/ReadVariableOp?
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/gru_cell/unstack?
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru/gru_cell/MatMul/ReadVariableOp?
gru/gru_cell/MatMulMatMulgru/strided_slice_1:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd?
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/gru_cell/split/split_dim?
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/gru_cell/split?
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$gru/gru_cell/MatMul_1/ReadVariableOp?
gru/gru_cell/MatMul_1MatMulzeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_1?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_1}
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru/gru_cell/Const?
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru/gru_cell/split_1/split_dim?
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/gru_cell/split_1?
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add?
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid?
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_1?
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid_1?
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul?
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_2y
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Relu?
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_1m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/sub?
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_2?
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_3?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0zeros:output:0gru/strided_slice:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	* 
bodyR
gru_while_body_10388* 
condR
gru_while_cond_10387*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:K??????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????K?2
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freeq
dense/Tensordot/ShapeShapegru/transpose_1:y:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposegru/transpose_1:y:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????K?2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/Tensordot/MatMul}
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????K?2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????K?2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while:P L
'
_output_shapes
:?????????K
!
_user_specified_name	input_1
?
?
#query_generator_gru_while_cond_9614D
@query_generator_gru_while_query_generator_gru_while_loop_counterJ
Fquery_generator_gru_while_query_generator_gru_while_maximum_iterations)
%query_generator_gru_while_placeholder+
'query_generator_gru_while_placeholder_1+
'query_generator_gru_while_placeholder_2D
@query_generator_gru_while_less_query_generator_gru_strided_sliceZ
Vquery_generator_gru_while_query_generator_gru_while_cond_9614___redundant_placeholder0Z
Vquery_generator_gru_while_query_generator_gru_while_cond_9614___redundant_placeholder1Z
Vquery_generator_gru_while_query_generator_gru_while_cond_9614___redundant_placeholder2Z
Vquery_generator_gru_while_query_generator_gru_while_cond_9614___redundant_placeholder3&
"query_generator_gru_while_identity
?
query_generator/gru/while/LessLess%query_generator_gru_while_placeholder@query_generator_gru_while_less_query_generator_gru_strided_slice*
T0*
_output_shapes
: 2 
query_generator/gru/while/Less?
"query_generator/gru/while/IdentityIdentity"query_generator/gru/while/Less:z:0*
T0
*
_output_shapes
: 2$
"query_generator/gru/while/Identity"Q
"query_generator_gru_while_identity+query_generator/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?
?
gru_while_cond_10199$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2$
 gru_while_less_gru_strided_slice;
7gru_while_gru_while_cond_10199___redundant_placeholder0;
7gru_while_gru_while_cond_10199___redundant_placeholder1;
7gru_while_gru_while_cond_10199___redundant_placeholder2;
7gru_while_gru_while_cond_10199___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder gru_while_less_gru_strided_slice*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
C__inference_embedding_layer_call_and_return_conditional_losses_8676

inputs	(
embedding_lookup_8670:	?d
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_8670inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/8670*+
_output_shapes
:?????????Kd*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/8670*+
_output_shapes
:?????????Kd2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????Kd2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????Kd2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????K: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????K
 
_user_specified_nameinputs
ˁ
?
!__inference__traced_restore_11690
file_prefixc
Ystring_lookup_1_index_table_table_restore_lookuptableimportv2_string_lookup_1_index_table:	 _
Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_table: H
5assignvariableop_query_generator_embedding_embeddings:	?dC
/assignvariableop_1_query_generator_dense_kernel:
??<
-assignvariableop_2_query_generator_dense_bias:	?&
assignvariableop_3_adam_iter:	 (
assignvariableop_4_adam_beta_1: (
assignvariableop_5_adam_beta_2: '
assignvariableop_6_adam_decay: /
%assignvariableop_7_adam_learning_rate: I
6assignvariableop_8_query_generator_gru_gru_cell_kernel:	d?T
@assignvariableop_9_query_generator_gru_gru_cell_recurrent_kernel:
??H
5assignvariableop_10_query_generator_gru_gru_cell_bias:	?#
assignvariableop_11_total: #
assignvariableop_12_count: R
?assignvariableop_13_adam_query_generator_embedding_embeddings_m:	?dK
7assignvariableop_14_adam_query_generator_dense_kernel_m:
??D
5assignvariableop_15_adam_query_generator_dense_bias_m:	?Q
>assignvariableop_16_adam_query_generator_gru_gru_cell_kernel_m:	d?\
Hassignvariableop_17_adam_query_generator_gru_gru_cell_recurrent_kernel_m:
??O
<assignvariableop_18_adam_query_generator_gru_gru_cell_bias_m:	?R
?assignvariableop_19_adam_query_generator_embedding_embeddings_v:	?dK
7assignvariableop_20_adam_query_generator_dense_kernel_v:
??D
5assignvariableop_21_adam_query_generator_dense_bias_v:	?Q
>assignvariableop_22_adam_query_generator_gru_gru_cell_kernel_v:	d?\
Hassignvariableop_23_adam_query_generator_gru_gru_cell_recurrent_kernel_v:
??O
<assignvariableop_24_adam_query_generator_gru_gru_cell_bias_v:	?
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?=string_lookup_1_index_table_table_restore/LookupTableImportV2?;string_lookup_index_table_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B,chars_from_ids/_table/.ATTRIBUTES/table-keysB.chars_from_ids/_table/.ATTRIBUTES/table-valuesB,ids_from_chars/_table/.ATTRIBUTES/table-keysB.ids_from_chars/_table/.ATTRIBUTES/table-valuesB5model/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB-model/dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB+model/dense/bias/.ATTRIBUTES/VARIABLE_VALUEB/model/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB1model/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB1model/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB0model/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/3/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2			2
	RestoreV2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_1_index_table_table_restore_lookuptableimportv2_string_lookup_1_index_tableRestoreV2:tensors:0RestoreV2:tensors:1*	
Tin0	*

Tout0*.
_class$
" loc:@string_lookup_1_index_table*
_output_shapes
 2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2?
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_tableRestoreV2:tensors:2RestoreV2:tensors:3*	
Tin0*

Tout0	*,
_class"
 loc:@string_lookup_index_table*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2g
IdentityIdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp5assignvariableop_query_generator_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_query_generator_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_query_generator_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5l

Identity_6IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6l

Identity_7IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7l

Identity_8IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_query_generator_gru_gru_cell_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8l

Identity_9IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp@assignvariableop_9_query_generator_gru_gru_cell_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_query_generator_gru_gru_cell_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp?assignvariableop_13_adam_query_generator_embedding_embeddings_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_adam_query_generator_dense_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp5assignvariableop_15_adam_query_generator_dense_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp>assignvariableop_16_adam_query_generator_gru_gru_cell_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpHassignvariableop_17_adam_query_generator_gru_gru_cell_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp<assignvariableop_18_adam_query_generator_gru_gru_cell_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp?assignvariableop_19_adam_query_generator_embedding_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp7assignvariableop_20_adam_query_generator_dense_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_query_generator_dense_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_query_generator_gru_gru_cell_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpHassignvariableop_23_adam_query_generator_gru_gru_cell_recurrent_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp<assignvariableop_24_adam_query_generator_gru_gru_cell_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp>^string_lookup_1_index_table_table_restore/LookupTableImportV2<^string_lookup_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25?
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9>^string_lookup_1_index_table_table_restore/LookupTableImportV2<^string_lookup_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92~
=string_lookup_1_index_table_table_restore/LookupTableImportV2=string_lookup_1_index_table_table_restore/LookupTableImportV22z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:40
.
_class$
" loc:@string_lookup_1_index_table:2.
,
_class"
 loc:@string_lookup_index_table
?E
?
while_body_8748
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?B
/while_gru_cell_matmul_readvariableop_resource_0:	d?E
1while_gru_cell_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?@
-while_gru_cell_matmul_readvariableop_resource:	d?C
/while_gru_cell_matmul_1_readvariableop_resource:
????$while/gru_cell/MatMul/ReadVariableOp?&while/gru_cell/MatMul_1/ReadVariableOp?while/gru_cell/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02&
$while/gru_cell/MatMul/ReadVariableOp?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
while/gru_cell/split/split_dim?
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split?
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp?
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell/Const?
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell/split_1/split_dim?
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split_1?
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Relu?
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?	
?
__inference_restore_fn_11451
restored_tensors_0	
restored_tensors_1N
Jstring_lookup_1_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_1_index_table_table_restore/LookupTableImportV2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_1_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0	*

Tout0*
_output_shapes
 2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_1_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2~
=string_lookup_1_index_table_table_restore/LookupTableImportV2=string_lookup_1_index_table_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
??
?
I__inference_query_generator_layer_call_and_return_conditional_losses_9939

inputs	2
embedding_embedding_lookup_9754:	?d7
$gru_gru_cell_readvariableop_resource:	?>
+gru_gru_cell_matmul_readvariableop_resource:	d?A
-gru_gru_cell_matmul_1_readvariableop_resource:
??;
'dense_tensordot_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?embedding/embedding_lookup?"gru/gru_cell/MatMul/ReadVariableOp?$gru/gru_cell/MatMul_1/ReadVariableOp?gru/gru_cell/ReadVariableOp?	gru/while?
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_9754inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/9754*+
_output_shapes
:?????????Kd*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/9754*+
_output_shapes
:?????????Kd2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????Kd2'
%embedding/embedding_lookup/Identity_1l
ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0gru/transpose/perm:output:0*
T0*+
_output_shapes
:K?????????d2
gru/transposeW
	gru/ShapeShapegru/transpose:y:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/transpose:y:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru/strided_slice_1?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru/gru_cell/ReadVariableOp?
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/gru_cell/unstack?
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru/gru_cell/MatMul/ReadVariableOp?
gru/gru_cell/MatMulMatMulgru/strided_slice_1:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd?
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/gru_cell/split/split_dim?
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/gru_cell/split?
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$gru/gru_cell/MatMul_1/ReadVariableOp?
gru/gru_cell/MatMul_1MatMulzeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_1?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_1}
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru/gru_cell/Const?
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru/gru_cell/split_1/split_dim?
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/gru_cell/split_1?
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add?
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid?
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_1?
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid_1?
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul?
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_2y
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Relu?
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_1m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/sub?
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_2?
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_3?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0zeros:output:0gru/strided_slice:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
gru_while_body_9824*
condR
gru_while_cond_9823*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:K??????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????K?2
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freeq
dense/Tensordot/ShapeShapegru/transpose_1:y:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposegru/transpose_1:y:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????K?2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/Tensordot/MatMul}
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????K?2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????K?2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while:O K
'
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
/__inference_query_generator_layer_call_fn_10554

inputs	
unknown:	?d
	unknown_0:	?
	unknown_1:	d?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????K?*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_query_generator_layer_call_and_return_conditional_losses_91432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????K
 
_user_specified_nameinputs
?E
?
while_body_8982
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?B
/while_gru_cell_matmul_readvariableop_resource_0:	d?E
1while_gru_cell_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?@
-while_gru_cell_matmul_readvariableop_resource:	d?C
/while_gru_cell_matmul_1_readvariableop_resource:
????$while/gru_cell/MatMul/ReadVariableOp?&while/gru_cell/MatMul_1/ReadVariableOp?while/gru_cell/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02&
$while/gru_cell/MatMul/ReadVariableOp?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
while/gru_cell/split/split_dim?
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split?
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp?
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell/Const?
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell/split_1/split_dim?
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split_1?
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Relu?
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
gru_while_cond_10011$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2$
 gru_while_less_gru_strided_slice;
7gru_while_gru_while_cond_10011___redundant_placeholder0;
7gru_while_gru_while_cond_10011___redundant_placeholder1;
7gru_while_gru_while_cond_10011___redundant_placeholder2;
7gru_while_gru_while_cond_10011___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder gru_while_less_gru_strided_slice*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?E
?
while_body_10655
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?B
/while_gru_cell_matmul_readvariableop_resource_0:	d?E
1while_gru_cell_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?@
-while_gru_cell_matmul_readvariableop_resource:	d?C
/while_gru_cell_matmul_1_readvariableop_resource:
????$while/gru_cell/MatMul/ReadVariableOp?&while/gru_cell/MatMul_1/ReadVariableOp?while/gru_cell/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02&
$while/gru_cell/MatMul/ReadVariableOp?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
while/gru_cell/split/split_dim?
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split?
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp?
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell/Const?
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell/split_1/split_dim?
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split_1?
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Relu?
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_query_generator_layer_call_fn_10571
input_1	
unknown:	?d
	unknown_0:	?
	unknown_1:	d?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????K?*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_query_generator_layer_call_and_return_conditional_losses_91432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????K
!
_user_specified_name	input_1
?

?
#__inference_gru_layer_call_fn_11235

inputs
initial_state_0
unknown:	?
	unknown_0:	d?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????K?:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_88382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????K?2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????Kd:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????Kd
 
_user_specified_nameinputs:YU
(
_output_shapes
:??????????
)
_user_specified_nameinitial_state/0
?O
?
=__inference_gru_layer_call_and_return_conditional_losses_8838

inputs
initial_state3
 gru_cell_readvariableop_resource:	?:
'gru_cell_matmul_readvariableop_resource:	d?=
)gru_cell_matmul_1_readvariableop_resource:
??
identity

identity_1??gru_cell/MatMul/ReadVariableOp? gru_cell/MatMul_1/ReadVariableOp?gru_cell/ReadVariableOp?whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:K?????????d2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
gru_cell/MatMul/ReadVariableOp?
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split/split_dim?
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split?
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp?
gru_cell/MatMul_1MatMulinitial_state(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1u
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell/Const?
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split_1/split_dim?
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split_1?
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Relu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0initial_state*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub?
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_statestrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_8748*
condR
while_cond_8747*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:K??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????K?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*,
_output_shapes
:?????????K?2

Identity?

Identity_1Identitywhile:output:4^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????Kd:??????????: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????Kd
 
_user_specified_nameinputs:WS
(
_output_shapes
:??????????
'
_user_specified_nameinitial_state
??
?
J__inference_query_generator_layer_call_and_return_conditional_losses_10315
input_1	3
 embedding_embedding_lookup_10130:	?d7
$gru_gru_cell_readvariableop_resource:	?>
+gru_gru_cell_matmul_readvariableop_resource:	d?A
-gru_gru_cell_matmul_1_readvariableop_resource:
??;
'dense_tensordot_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?embedding/embedding_lookup?"gru/gru_cell/MatMul/ReadVariableOp?$gru/gru_cell/MatMul_1/ReadVariableOp?gru/gru_cell/ReadVariableOp?	gru/while?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_10130input_1",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/10130*+
_output_shapes
:?????????Kd*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/10130*+
_output_shapes
:?????????Kd2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????Kd2'
%embedding/embedding_lookup/Identity_1l
ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0gru/transpose/perm:output:0*
T0*+
_output_shapes
:K?????????d2
gru/transposeW
	gru/ShapeShapegru/transpose:y:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/transpose:y:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
gru/strided_slice_1?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru/gru_cell/ReadVariableOp?
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/gru_cell/unstack?
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02$
"gru/gru_cell/MatMul/ReadVariableOp?
gru/gru_cell/MatMulMatMulgru/strided_slice_1:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd?
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/gru_cell/split/split_dim?
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/gru_cell/split?
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$gru/gru_cell/MatMul_1/ReadVariableOp?
gru/gru_cell/MatMul_1MatMulzeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_1?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_1}
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru/gru_cell/Const?
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru/gru_cell/split_1/split_dim?
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/gru_cell/split_1?
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add?
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid?
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_1?
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid_1?
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul?
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_2y
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Relu?
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_1m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/sub?
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_2?
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_3?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0zeros:output:0gru/strided_slice:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	* 
bodyR
gru_while_body_10200* 
condR
gru_while_cond_10199*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:K??????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????K?2
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freeq
dense/Tensordot/ShapeShapegru/transpose_1:y:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposegru/transpose_1:y:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????K?2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/Tensordot/MatMul}
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????K?2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????K?2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while:P L
'
_output_shapes
:?????????K
!
_user_specified_name	input_1
?

?
(__inference_gru_cell_layer_call_fn_11410

inputs
states_0
unknown:	?
	unknown_0:	d?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_81402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
#query_generator_gru_while_cond_9368D
@query_generator_gru_while_query_generator_gru_while_loop_counterJ
Fquery_generator_gru_while_query_generator_gru_while_maximum_iterations)
%query_generator_gru_while_placeholder+
'query_generator_gru_while_placeholder_1+
'query_generator_gru_while_placeholder_2D
@query_generator_gru_while_less_query_generator_gru_strided_sliceZ
Vquery_generator_gru_while_query_generator_gru_while_cond_9368___redundant_placeholder0Z
Vquery_generator_gru_while_query_generator_gru_while_cond_9368___redundant_placeholder1Z
Vquery_generator_gru_while_query_generator_gru_while_cond_9368___redundant_placeholder2Z
Vquery_generator_gru_while_query_generator_gru_while_cond_9368___redundant_placeholder3&
"query_generator_gru_while_identity
?
query_generator/gru/while/LessLess%query_generator_gru_while_placeholder@query_generator_gru_while_less_query_generator_gru_strided_slice*
T0*
_output_shapes
: 2 
query_generator/gru/while/Less?
"query_generator/gru/while/IdentityIdentity"query_generator/gru/while/Less:z:0*
T0
*
_output_shapes
: 2$
"query_generator/gru/while/Identity"Q
"query_generator_gru_while_identity+query_generator/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?K
?
gru_while_body_10388$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2!
gru_while_gru_strided_slice_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	?F
3gru_while_gru_cell_matmul_readvariableop_resource_0:	d?I
5gru_while_gru_cell_matmul_1_readvariableop_resource_0:
??
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_gru_strided_slice]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	?D
1gru_while_gru_cell_matmul_readvariableop_resource:	d?G
3gru_while_gru_cell_matmul_1_readvariableop_resource:
????(gru/while/gru_cell/MatMul/ReadVariableOp?*gru/while/gru_cell/MatMul_1/ReadVariableOp?!gru/while/gru_cell/ReadVariableOp?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/while/gru_cell/unstack?
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02*
(gru/while/gru_cell/MatMul/ReadVariableOp?
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd?
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru/while/gru_cell/split/split_dim?
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/while/gru_cell/split?
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*gru/while/gru_cell/MatMul_1/ReadVariableOp?
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_1?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_1?
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru/while/gru_cell/Const?
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru/while/gru_cell/split_1/split_dim?
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru/while/gru_cell/split_1?
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add?
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid?
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_1?
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid_1?
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul?
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_2?
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Relu?
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_1y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_2?
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_3?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1?
gru/while/IdentityIdentitygru/while/add_1:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_1?
gru/while/Identity_2Identitygru/while/add:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
gru/while/Identity_4"l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"<
gru_while_gru_strided_slicegru_while_gru_strided_slice_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_8152
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8152___redundant_placeholder02
.while_while_cond_8152___redundant_placeholder12
.while_while_cond_8152___redundant_placeholder22
.while_while_cond_8152___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
D__inference_embedding_layer_call_and_return_conditional_losses_10580

inputs	)
embedding_lookup_10574:	?d
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_10574inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*)
_class
loc:@embedding_lookup/10574*+
_output_shapes
:?????????Kd*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/10574*+
_output_shapes
:?????????Kd2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????Kd2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????Kd2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????K: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????K
 
_user_specified_nameinputs
?	
?
#__inference_gru_layer_call_fn_11208
inputs_0
unknown:	?
	unknown_0:	d?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:???????????????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_82182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????d
"
_user_specified_name
inputs/0
?>
?
=__inference_gru_layer_call_and_return_conditional_losses_8420

inputs 
gru_cell_8343:	? 
gru_cell_8345:	d?!
gru_cell_8347:
??
identity

identity_1?? gru_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_2?
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_8343gru_cell_8345gru_cell_8347*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_82862"
 gru_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_8343gru_cell_8345gru_cell_8347*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_8355*
condR
while_cond_8354*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0!^gru_cell/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identitywhile:output:4!^gru_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????d: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs
?
,
__inference__destroyer_11318
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
#__inference_gru_layer_call_fn_11249

inputs
initial_state_0
unknown:	?
	unknown_0:	d?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????K?:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_90722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????K?2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????Kd:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????Kd
 
_user_specified_nameinputs:YU
(
_output_shapes
:??????????
)
_user_specified_nameinitial_state/0
?
?
__inference_save_fn_11470
checkpoint_keyY
Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2J
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityOstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:keys:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity	add_1:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityQstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:values:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
while_cond_10958
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_10958___redundant_placeholder03
/while_while_cond_10958___redundant_placeholder13
/while_while_cond_10958___redundant_placeholder23
/while_while_cond_10958___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?O
?
=__inference_gru_layer_call_and_return_conditional_losses_9072

inputs
initial_state3
 gru_cell_readvariableop_resource:	?:
'gru_cell_matmul_readvariableop_resource:	d?=
)gru_cell_matmul_1_readvariableop_resource:
??
identity

identity_1??gru_cell/MatMul/ReadVariableOp? gru_cell/MatMul_1/ReadVariableOp?gru_cell/ReadVariableOp?whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:K?????????d2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
gru_cell/MatMul/ReadVariableOp?
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split/split_dim?
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split?
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp?
gru_cell/MatMul_1MatMulinitial_state(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1u
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell/Const?
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split_1/split_dim?
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split_1?
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Relu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0initial_state*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub?
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_statestrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_8982*
condR
while_cond_8981*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:K??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????K?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*,
_output_shapes
:?????????K?2

Identity?

Identity_1Identitywhile:output:4^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????Kd:??????????: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????Kd
 
_user_specified_nameinputs:WS
(
_output_shapes
:??????????
'
_user_specified_nameinitial_state
?[
?
>__inference_gru_layer_call_and_return_conditional_losses_10745
inputs_03
 gru_cell_readvariableop_resource:	?:
'gru_cell_matmul_readvariableop_resource:	d?=
)gru_cell_matmul_1_readvariableop_resource:
??
identity

identity_1??gru_cell/MatMul/ReadVariableOp? gru_cell/MatMul_1/ReadVariableOp?gru_cell/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_2?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
gru_cell/MatMul/ReadVariableOp?
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split/split_dim?
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split?
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp?
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1u
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell/Const?
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split_1/split_dim?
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split_1?
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Relu?
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub?
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10655*
condR
while_cond_10654*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identitywhile:output:4^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????d: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????d
"
_user_specified_name
inputs/0
?O
?
>__inference_gru_layer_call_and_return_conditional_losses_11195

inputs
initial_state_03
 gru_cell_readvariableop_resource:	?:
'gru_cell_matmul_readvariableop_resource:	d?=
)gru_cell_matmul_1_readvariableop_resource:
??
identity

identity_1??gru_cell/MatMul/ReadVariableOp? gru_cell/MatMul_1/ReadVariableOp?gru_cell/ReadVariableOp?whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:K?????????d2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
gru_cell/MatMul/ReadVariableOp?
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split/split_dim?
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split?
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp?
gru_cell/MatMul_1MatMulinitial_state_0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1u
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell/Const?
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split_1/split_dim?
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell/split_1?
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Relu?
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0initial_state_0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub?
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_state_0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_11105*
condR
while_cond_11104*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:K??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????K?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*,
_output_shapes
:?????????K?2

Identity?

Identity_1Identitywhile:output:4^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????Kd:??????????: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????Kd
 
_user_specified_nameinputs:YU
(
_output_shapes
:??????????
)
_user_specified_nameinitial_state/0
?E
?
while_body_11105
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?B
/while_gru_cell_matmul_readvariableop_resource_0:	d?E
1while_gru_cell_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?@
-while_gru_cell_matmul_readvariableop_resource:	d?C
/while_gru_cell_matmul_1_readvariableop_resource:
????$while/gru_cell/MatMul/ReadVariableOp?&while/gru_cell/MatMul_1/ReadVariableOp?while/gru_cell/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02&
$while/gru_cell/MatMul/ReadVariableOp?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
while/gru_cell/split/split_dim?
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split?
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp?
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell/Const?
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell/split_1/split_dim?
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell/split_1?
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Relu?
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_query_generator_layer_call_and_return_conditional_losses_9143

inputs	!
embedding_9113:	?d
gru_9129:	?
gru_9131:	d?
gru_9133:
??

dense_9137:
??

dense_9139:	?
identity??dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?gru/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_9113*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????Kd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_86762#
!embedding/StatefulPartitionedCallh
ShapeShape*embedding/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zeros?
gru/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0zeros:output:0gru_9129gru_9131gru_9133*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????K?:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_90722
gru/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0
dense_9137
dense_9139*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????K?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_88772
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????K?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:O K
'
_output_shapes
:?????????K
 
_user_specified_nameinputs
?	
?
__inference_restore_fn_11478
restored_tensors_0
restored_tensors_1	L
Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handle
identity??;string_lookup_index_table_table_restore/LookupTableImportV2?
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0<^string_lookup_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
Q
__inference__creator_11293
identity:	 ??string_lookup_1_index_table?
string_lookup_1_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name
table_15*
value_dtype02
string_lookup_1_index_table?
IdentityIdentity*string_lookup_1_index_table:table_handle:0^string_lookup_1_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2:
string_lookup_1_index_tablestring_lookup_1_index_table
?
?
while_cond_8354
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8354___redundant_placeholder02
.while_while_cond_8354___redundant_placeholder12
.while_while_cond_8354___redundant_placeholder22
.while_while_cond_8354___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
gru_while_cond_10387$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2$
 gru_while_less_gru_strided_slice;
7gru_while_gru_while_cond_10387___redundant_placeholder0;
7gru_while_gru_while_cond_10387___redundant_placeholder1;
7gru_while_gru_while_cond_10387___redundant_placeholder2;
7gru_while_gru_while_cond_10387___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder gru_while_less_gru_strided_slice*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_11396

inputs
states_0*
readvariableop_resource:	?1
matmul_readvariableop_resource:	d?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3?
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?!
?
while_body_8153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_gru_cell_8175_0:	?(
while_gru_cell_8177_0:	d?)
while_gru_cell_8179_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_gru_cell_8175:	?&
while_gru_cell_8177:	d?'
while_gru_cell_8179:
????&while/gru_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????d*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_8175_0while_gru_cell_8177_0while_gru_cell_8179_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_81402(
&while/gru_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4",
while_gru_cell_8175while_gru_cell_8175_0",
while_gru_cell_8177while_gru_cell_8177_0",
while_gru_cell_8179while_gru_cell_8179_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: "?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?
	model
chars_from_ids
ids_from_chars
	keras_api

signatures
\generate_one_step"?
_tf_keras_model?{"name": "one_step", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "OneStep", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": null, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "OneStep"}}
?	
	embedding
rnn
	dense
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"?
_tf_keras_model?{"name": "query_generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QueryGenerator", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 75]}, "int64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QueryGenerator"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
state_variables

_table
	keras_api"?
_tf_keras_layer?{"name": "string_lookup_1", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": true, "must_restore_from_config": true, "class_name": "StringLookup", "config": {"name": "string_lookup_1", "trainable": true, "dtype": "string", "invert": true, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 427, "vocabulary": null, "encoding": "utf-8"}}
?
state_variables

_table
	keras_api"?
_tf_keras_layer?{"name": "string_lookup", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": true, "must_restore_from_config": true, "class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 427, "vocabulary": null, "encoding": "utf-8"}}
"
_generic_user_object
"
signature_map
?

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"?
_tf_keras_layer?{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 427, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 75]}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"?

_tf_keras_rnn_layer?	{"name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": true, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [64, 75, 100]}}
?

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
*c&call_and_return_all_conditional_losses
d__call__"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 427, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 75, 128]}}
?
%iter

&beta_1

'beta_2
	(decay
)learning_ratemPmQ mR*mS+mT,mUvVvW vX*vY+vZ,v["
	optimizer
 "
trackable_list_wrapper
J
0
*1
+2
,3
4
 5"
trackable_list_wrapper
J
0
*1
+2
,3
4
 5"
trackable_list_wrapper
?

regularization_losses
	variables

-layers
.non_trainable_variables
/metrics
0layer_regularization_losses
1layer_metrics
trainable_variables
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
O
e_create_resource
f_initialize
g_destroy_resourceR Z
tablehi
"
_generic_user_object
 "
trackable_dict_wrapper
O
j_create_resource
k_initialize
l_destroy_resourceR Z
tablemn
"
_generic_user_object
7:5	?d2$query_generator/embedding/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
regularization_losses
	variables

2layers
3non_trainable_variables
4metrics
5layer_regularization_losses
6layer_metrics
trainable_variables
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
?

*kernel
+recurrent_kernel
,bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
*o&call_and_return_all_conditional_losses
p__call__"?
_tf_keras_layer?{"name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
?
regularization_losses
	variables

;states

<layers
=non_trainable_variables
>metrics
?layer_regularization_losses
@layer_metrics
trainable_variables
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
0:.
??2query_generator/dense/kernel
):'?2query_generator/dense/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
!regularization_losses
"	variables

Alayers
Bnon_trainable_variables
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
#trainable_variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
6:4	d?2#query_generator/gru/gru_cell/kernel
A:?
??2-query_generator/gru/gru_cell/recurrent_kernel
4:2	?2!query_generator/gru/gru_cell/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
'
F0"
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
5
*0
+1
,2"
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
?
7regularization_losses
8	variables

Glayers
Hnon_trainable_variables
Imetrics
Jlayer_regularization_losses
Klayer_metrics
9trainable_variables
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
?
	Ltotal
	Mcount
N	variables
O	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
<::	?d2+Adam/query_generator/embedding/embeddings/m
5:3
??2#Adam/query_generator/dense/kernel/m
.:,?2!Adam/query_generator/dense/bias/m
;:9	d?2*Adam/query_generator/gru/gru_cell/kernel/m
F:D
??24Adam/query_generator/gru/gru_cell/recurrent_kernel/m
9:7	?2(Adam/query_generator/gru/gru_cell/bias/m
<::	?d2+Adam/query_generator/embedding/embeddings/v
5:3
??2#Adam/query_generator/dense/kernel/v
.:,?2!Adam/query_generator/dense/bias/v
;:9	d?2*Adam/query_generator/gru/gru_cell/kernel/v
F:D
??24Adam/query_generator/gru/gru_cell/recurrent_kernel/v
9:7	?2(Adam/query_generator/gru/gru_cell/bias/v
?2?
"__inference_generate_one_step_9499
"__inference_generate_one_step_9745?
???
FullArgSpec'
args?
jself
jinputs
jstates
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_query_generator_layer_call_and_return_conditional_losses_9939
J__inference_query_generator_layer_call_and_return_conditional_losses_10127
J__inference_query_generator_layer_call_and_return_conditional_losses_10315
J__inference_query_generator_layer_call_and_return_conditional_losses_10503?
???
FullArgSpecC
args;?8
jself
jinputs
jstates
jreturn_state

jtraining
varargs
 
varkw
 
defaults?

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_query_generator_layer_call_fn_10520
/__inference_query_generator_layer_call_fn_10537
/__inference_query_generator_layer_call_fn_10554
/__inference_query_generator_layer_call_fn_10571?
???
FullArgSpecC
args;?8
jself
jinputs
jstates
jreturn_state

jtraining
varargs
 
varkw
 
defaults?

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_embedding_layer_call_and_return_conditional_losses_10580?
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
)__inference_embedding_layer_call_fn_10587?
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
?2?
>__inference_gru_layer_call_and_return_conditional_losses_10745
>__inference_gru_layer_call_and_return_conditional_losses_10903
>__inference_gru_layer_call_and_return_conditional_losses_11049
>__inference_gru_layer_call_and_return_conditional_losses_11195?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_gru_layer_call_fn_11208
#__inference_gru_layer_call_fn_11221
#__inference_gru_layer_call_fn_11235
#__inference_gru_layer_call_fn_11249?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_11279?
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
%__inference_dense_layer_call_fn_11288?
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
__inference__creator_11293?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_11298?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_11303?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_11443checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_11451restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?	
	?
?2?
__inference__creator_11308?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_11313?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_11318?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_11470checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_11478restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
?2?
C__inference_gru_cell_layer_call_and_return_conditional_losses_11357
C__inference_gru_cell_layer_call_and_return_conditional_losses_11396?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

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
(__inference_gru_cell_layer_call_fn_11410
(__inference_gru_cell_layer_call_fn_11424?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

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
	J
Const
J	
Const_1
J	
Const_26
__inference__creator_11293?

? 
? "? 6
__inference__creator_11308?

? 
? "? 8
__inference__destroyer_11303?

? 
? "? 8
__inference__destroyer_11318?

? 
? "? :
__inference__initializer_11298?

? 
? "? :
__inference__initializer_11313?

? 
? "? ?
@__inference_dense_layer_call_and_return_conditional_losses_11279f 4?1
*?'
%?"
inputs?????????K?
? "*?'
 ?
0?????????K?
? ?
%__inference_dense_layer_call_fn_11288Y 4?1
*?'
%?"
inputs?????????K?
? "??????????K??
D__inference_embedding_layer_call_and_return_conditional_losses_10580_/?,
%?"
 ?
inputs?????????K	
? ")?&
?
0?????????Kd
? 
)__inference_embedding_layer_call_fn_10587R/?,
%?"
 ?
inputs?????????K	
? "??????????Kd?
"__inference_generate_one_step_9499hq,*+ rs&?#
?
?
inputs

 
? "1?.
?
0
?
1???????????
"__inference_generate_one_step_9745uq,*+ rs<?9
2?/
?
inputs
?
states	?
? "(?%
?
0
?
1	??
C__inference_gru_cell_layer_call_and_return_conditional_losses_11357?,*+]?Z
S?P
 ?
inputs?????????d
(?%
#? 
states/0??????????
p 
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
C__inference_gru_cell_layer_call_and_return_conditional_losses_11396?,*+]?Z
S?P
 ?
inputs?????????d
(?%
#? 
states/0??????????
p
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
(__inference_gru_cell_layer_call_fn_11410?,*+]?Z
S?P
 ?
inputs?????????d
(?%
#? 
states/0??????????
p 
? "F?C
?
0??????????
#? 
?
1/0???????????
(__inference_gru_cell_layer_call_fn_11424?,*+]?Z
S?P
 ?
inputs?????????d
(?%
#? 
states/0??????????
p
? "F?C
?
0??????????
#? 
?
1/0???????????
>__inference_gru_layer_call_and_return_conditional_losses_10745?,*+O?L
E?B
4?1
/?,
inputs/0??????????????????d

 
p 

 
? "Z?W
P?M
+?(
0/0???????????????????
?
0/1??????????
? ?
>__inference_gru_layer_call_and_return_conditional_losses_10903?,*+O?L
E?B
4?1
/?,
inputs/0??????????????????d

 
p

 
? "Z?W
P?M
+?(
0/0???????????????????
?
0/1??????????
? ?
>__inference_gru_layer_call_and_return_conditional_losses_11049?,*+l?i
b?_
$?!
inputs?????????Kd

 
p 
/?,
*?'
initial_state/0??????????
? "Q?N
G?D
"?
0/0?????????K?
?
0/1??????????
? ?
>__inference_gru_layer_call_and_return_conditional_losses_11195?,*+l?i
b?_
$?!
inputs?????????Kd

 
p
/?,
*?'
initial_state/0??????????
? "Q?N
G?D
"?
0/0?????????K?
?
0/1??????????
? ?
#__inference_gru_layer_call_fn_11208?,*+O?L
E?B
4?1
/?,
inputs/0??????????????????d

 
p 

 
? "L?I
)?&
0???????????????????
?
1???????????
#__inference_gru_layer_call_fn_11221?,*+O?L
E?B
4?1
/?,
inputs/0??????????????????d

 
p

 
? "L?I
)?&
0???????????????????
?
1???????????
#__inference_gru_layer_call_fn_11235?,*+l?i
b?_
$?!
inputs?????????Kd

 
p 
/?,
*?'
initial_state/0??????????
? "C?@
 ?
0?????????K?
?
1???????????
#__inference_gru_layer_call_fn_11249?,*+l?i
b?_
$?!
inputs?????????Kd

 
p
/?,
*?'
initial_state/0??????????
? "C?@
 ?
0?????????K?
?
1???????????
J__inference_query_generator_layer_call_and_return_conditional_losses_10127q,*+ ;?8
1?.
 ?
inputs?????????K	

 
p 
p
? "*?'
 ?
0?????????K?
? ?
J__inference_query_generator_layer_call_and_return_conditional_losses_10315r,*+ <?9
2?/
!?
input_1?????????K	

 
p 
p 
? "*?'
 ?
0?????????K?
? ?
J__inference_query_generator_layer_call_and_return_conditional_losses_10503r,*+ <?9
2?/
!?
input_1?????????K	

 
p 
p
? "*?'
 ?
0?????????K?
? ?
I__inference_query_generator_layer_call_and_return_conditional_losses_9939q,*+ ;?8
1?.
 ?
inputs?????????K	

 
p 
p 
? "*?'
 ?
0?????????K?
? ?
/__inference_query_generator_layer_call_fn_10520e,*+ <?9
2?/
!?
input_1?????????K	

 
p 
p 
? "??????????K??
/__inference_query_generator_layer_call_fn_10537d,*+ ;?8
1?.
 ?
inputs?????????K	

 
p 
p 
? "??????????K??
/__inference_query_generator_layer_call_fn_10554d,*+ ;?8
1?.
 ?
inputs?????????K	

 
p 
p
? "??????????K??
/__inference_query_generator_layer_call_fn_10571e,*+ <?9
2?/
!?
input_1?????????K	

 
p 
p
? "??????????K?y
__inference_restore_fn_11451YK?H
A?>
?
restored_tensors_0	
?
restored_tensors_1
? "? y
__inference_restore_fn_11478YK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_11443?&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor	
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor?
__inference_save_fn_11470?&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	