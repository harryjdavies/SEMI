import torch

def SEMI(input_tensor,target_tensor,window_size,shift,step):
        
        # shift of 100 with step size 1 is from t-50 to t+49
        # shift of 50 with step size 2 is from t-50 to t+49
        # shift of 2 with step size of 10 is from t-20 to t+19 
        length_tensor = input_tensor.size(dim=-1)
        #number of times to perform the shifting MSE is the length of the sequence divided by the desired window size
        loop_size = int(length_tensor/window_size)
        shift_step = shift*step

        for n in range(loop_size):
            start_ind = int(n*window_size)
            end_ind = int(((n+1)*window_size))
            temp_input = input_tensor[:,:,start_ind:end_ind]
            switch_var = 0
            # this loop calculates MSE for each shift, for this specific window, and stores this value
            for m in range(int(shift)):
                m_step = m*step
                true_shift = int(m_step-(shift_step/2))
                start_ind_target = start_ind+true_shift
                end_ind_target = end_ind+true_shift
                # 3 conditions, if shift is before or after sequence then crop sequence
                if start_ind_target > -1 and end_ind_target < (length_tensor+1):
                    temp_target = target_tensor[:,:,start_ind_target:end_ind_target]
                    MSE = torch.mean(torch.square(temp_target-temp_input),dim=2)
                    if switch_var == 0:
                        MSE_temp = MSE
                        switch_var = 1
                    else:
                        MSE_temp = torch.cat((MSE_temp,MSE),dim=1)
                elif start_ind_target < 0 and end_ind_target < (length_tensor+1) and abs(start_ind_target)<window_size:
                    temp_target = target_tensor[:,:,0:end_ind_target]
                    input_start_ind_crop = -1*(window_size+start_ind_target) - 1
                    temp_input_crop = temp_input[:,:,input_start_ind_crop:-1]
                    MSE = torch.mean(torch.square(temp_target-temp_input_crop),dim=2)
                    if switch_var == 0:
                        MSE_temp = MSE
                        switch_var = 1
                    else:
                        MSE_temp = torch.cat((MSE_temp,MSE),dim=1)
                elif start_ind_target > -1 and end_ind_target > (length_tensor) and end_ind_target-length_tensor<window_size:
                    overlap = end_ind_target-length_tensor
                    temp_target = target_tensor[:,:,start_ind_target-1:-1]
                    input_end_ind_crop = -1*(overlap)
                    temp_input_crop = temp_input[:,:,0:input_end_ind_crop]
                    MSE = torch.mean(torch.square(temp_target-temp_input_crop),dim=2)
                    if switch_var == 0:
                        MSE_temp = MSE
                        switch_var = 1
                    else:
                        MSE_temp = torch.cat((MSE_temp,MSE),dim=1)
            # calculate the minimum of the MSEs stored for this window (best fit) and then store this.
            min_MSE_temp =   torch.mean(torch.min(MSE_temp,dim = 1)[0])
            if n > 0:
                min_MSE_store = torch.cat((min_MSE_store,min_MSE_temp.reshape(1)))
            else:
                min_MSE_store = min_MSE_temp.reshape(1)
        # calculate the mean of all the best fits across all windows, this is the final loss
        SEMI_loss = torch.mean(min_MSE_store)
        return SEMI_loss
    


# example implementation with a window size of 50, and a shift of 100 (t-49 to t+50):
#loss = SEMI(outputs, trainyT_seg,50,100)






