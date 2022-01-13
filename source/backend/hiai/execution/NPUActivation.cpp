    if(mType == 5){
        shared_ptr<hiai::op::PRelu> prelu(new hiai::op::PRelu(opName + "_prelu"));
        auto slopePtr = mOp->main_as_PRelu()->slope()->data();
        auto slopeSize = mOp->main_as_PRelu()->slope()->size();

        mConst_w = ge::op::Const(opName + "_w_const");
        {
            ge::TensorDesc fdesc(ge::Shape({1, slopeSize, 1, 1}), ge::FORMAT_NCHW,
                                ge::DT_FLOAT); // in o h w ?
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)slopePtr, slopeSize * sizeof(float));
            mConst_w.set_attr_value(filter);
        }

        (*prelu)
            .set_input_x(*xOp.get()).set_input_weight(mConst_w);
        mNpuBackend->setOutputOps(mOp, {prelu}, outputs);
    }

        if (op->type() == OpType_ReLU6) {
            return new NPUActivation(backend, op, inputs, outputs, 14);
        }else if (op->type() == OpType_Sigmoid) {
            return new NPUActivation(backend, op, inputs, outputs, 0);
        }else if (op->type() == OpType_PReLU) {
            return new NPUActivation(backend, op, inputs, outputs, 5);
        }else if (op->type() == OpType_TanH) {
            return new NPUActivation(backend, op, inputs, outputs, 2);
        }

NPUCreatorRegister<ActivationCreator> __relu6_op(OpType_ReLU6);
NPUCreatorRegister<ActivationCreator> __sigmoid_op(OpType_Sigmoid);
NPUCreatorRegister<ActivationCreator> __prelu_op(OpType_PReLU);
NPUCreatorRegister<ActivationCreator> __tanh_op(OpType_TanH);