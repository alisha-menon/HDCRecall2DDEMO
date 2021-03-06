
function message = HD_functions_HDrorb
  assignin('base','genBRandomHV', @genBRandomHV); 
  assignin('base','projBRandomHV', @projBRandomHV); 
  assignin('base','initItemMemories', @initItemMemories);
  assignin('base','projItemMemeory', @projItemMemeory); 
  assignin('base','computeNgramproj', @computeNgramproj); 
  assignin('base','hdcrorbtrain', @hdcrorbtrain); 
  assignin('base','hdcrorbpredict', @hdcrorbpredict);   
  assignin('base','genRandomHV', @genRandomHV); 
  assignin('base','downSampling', @downSampling);
  assignin('base','genTrainData', @genTrainData);
  assignin('base','lookupItemMemeory', @lookupItemMemeory);
  assignin('base','hamming', @hamming);
  message='Importing all HD functions';
end


function [CiMC, iMC, CiMR, iMR] = initItemMemories (D, MAXLC, channelsC, MAXLR, channelsR)
%
% DESCRIPTION    : initialize the item Memory  
%
% INPUTS:
%   D            : Dimension of vectors
%   MAXLC        : # of vectors in CiM for conditions
%   channelsC    : Number of acquisition channels for conditions
%   MAXLR        : # of vectors in CiM for results
%   channelsR    : Number of acquisition channels for results

% OUTPUTS:
%   iMC          : item memory for IDs of channels for conditions
%   CiMC         : continious item memory for value of a channel for
%   conditions
%   iMR          : item memory for IDs of channels for results
%   CiMR         : continious item memory for value of a channel for
%   results
 
    CiMC = containers.Map ('KeyType','double','ValueType','any');
    iMC  = containers.Map ('KeyType','double','ValueType','any');
    CiMR = containers.Map ('KeyType','double','ValueType','any');
    iMR  = containers.Map ('KeyType','double','ValueType','any');
    rng('default');
    rng(1);
    
    if channelsC == channelsR
        for i = 1 : channelsC
            iMC(i) = genRandomHV (D);
            iMR(i) = genRandomHV (D);
        end
    else
        for i = 1 : channelsC
            iMC(i) = genRandomHV (D);
        end

        for i = 1 : channelsR
            iMR(i) = genRandomHV (D);
        end
    end

    if MAXLC==MAXLR
        initHVC = genRandomHV (D);
        currentHVC = initHVC;
        initHVR = genRandomHV (D);
        currentHVR = initHVR;
        for i = 0:1:MAXLC-1
            CiMC(i) = currentHVC; 
            currentHVC = genRandomHV(D);
            CiMR(i) = currentHVR;
            currentHVR = genRandomHV(D);
        end
    else
        initHV = genRandomHV (D);
        currentHV = initHV;
        for i = 0:1:MAXLC-1
            CiMC(i) = currentHV; 
            currentHV = genRandomHV(D);
        end
        initHV = genRandomHV (D);
        currentHV = initHV;
        for i = 0:1:MAXLR-1
            CiMR(i) = currentHV; 
            currentHV = genRandomHV(D);
        end
    end
end

function randomHV = genRandomHV(D)
%
% DESCRIPTION   : generate a random vector with zero mean 
%
% INPUTS:
%   D           : Dimension of vectors
% OUTPUTS:
%   randomHV    : generated random vector

    if mod(D,2)
        disp ('Dimension is odd!!');
    else
        randomIndex = randperm (D);
        randomHV (randomIndex(1 : D/2)) = 1;
        randomHV (randomIndex(D/2+1 : D)) = 0;
       
    end
end
  
function [prog_HV] = hdcrorbtrain (few_all, length_training, features_condition, features_result, chAMcondition, chAMresult, iMchcondition, iMchresult, D, channels_condition, channels_result)
%
% DESCRIPTION          : train a program memory based on specific condition,result pairs
%
% INPUTS:
%   length_training    : # of training data samples
%   features_condition : condition feature training data
%   features_result    : result feature training data
%   chAMcondition      : cont. item memory for condition data
%   chAMresult         : cont. item memory for result data
%   iMchcondition      : item memory for condition channels
%   iMchresult         : item memory for result channels
%   D                  : Dimension of vectors
%   precision          : Precision determines size of CiM and also actuator quantization of returned values
%   channels_condition : number of condition channels
%   channels_results   : number of result channels
%
% OUTPUTS:
%   prog_HV            : Trained program memory
%   result_AM          : AM for result vectors


    condition_vectorlist = zeros (channels_condition, D);
    result_vectorlist = zeros (channels_result, D);
    prog_HVlist = zeros(length_training,D);
    
    i = 1;
    while i <= length_training
        for x = 1:1:channels_condition
            %[CiM_condition] = lookupItemMemeory (chAMcondition, features_condition(i,x));
            condition_vectorlist(x,:) = xor(chAMcondition(features_condition(i,x)) , iMchcondition(x));
        end
        if channels_condition > 1
            if (mod(channels_condition, 2) == 1)
                condition_vector = mode(condition_vectorlist);
            else
                %condition_vector = mode([condition_vectorlist; circshift(condition_vectorlist(1,:), [1,1])]);
                if (few_all == 0)
                    extra_condition_vector = xor(condition_vectorlist(1,:),condition_vectorlist(2,:));
                else
                    extra_condition_vector = condition_vectorlist(1,:);
                    for y = 2:1:channels_condition
                        extra_condition_vector = xor(extra_condition_vector,condition_vectorlist(y,:));
                    end
                end
                extra_condition_vector = circshift(extra_condition_vector, [1,1]);
                condition_vector = mode([condition_vectorlist; extra_condition_vector]);
            end  
        else
            condition_vector = condition_vectorlist(1,:);
        end
        for m = 1:1:channels_result
            %[CiM_result] = lookupItemMemeory (chAMresult, features_result(i,m));
            result_vectorlist(m,:) = xor(chAMresult(features_result(i,m)) , iMchresult(m));
        end
        if channels_result > 1
            if (mod(channels_result, 2) == 1)
                result_vector = mode(result_vectorlist);
            else
                if (few_all == 0)
                    extra_result_vector = xor(result_vectorlist(1,:), result_vectorlist(2,:));
                else
                    extra_result_vector = result_vectorlist(1,:);
                    for y = 2:1:channels_result
                        extra_result_vector = xor(extra_result_vector,result_vectorlist(y,:));
                    end
                end
                extra_result_vector = circshift(extra_result_vector, [1,1]);
                result_vector = mode([result_vectorlist; extra_result_vector]);
            end
        else
            result_vector = result_vectorlist(1,:);
        end
        protected_condition = circshift(condition_vector, [1,1]);
        prog_HVlist(i,:) = xor(protected_condition,result_vector);
        i = i + 1;
    end
    if (mod(i-1, 2) == 1)
        prog_HV = mode(prog_HVlist(1:i-1,:));
    else
        prog_HV = mode([prog_HVlist(1:i-1,:); genRandomHV(D)]);
    end
end

function [actuator_values] = hdcrorbpredict (few_all, length_recall, prog_HV, features_condition, chAMcondition, chAMresult, iMchcondition, iMchresult, D, channels_condition, channels_result)
%
% DESCRIPTION   : test accuracy based on input testing data
%
% INPUTS:
%   few_all      : Selects method of bundling extra vectors for even # of
%   vector cases
%   length_recall: # of recall samples
%   prog_HV      : Program hypervector
%   features_condition      : recall condition data
%   chAM           : Trained associative memory
%   CiM          : Cont. item memory (no use)
%   iM           : item memory
%   D            : Dimension of vectors
%   N            : size of n-gram, i.e., window size 
%
% OUTPUTS:
%   accuracy     : classification accuracy for all situations
%   accExcTrnz   : classification accuracy excluding the transitions between gestutes
%

    actuator_values = zeros(length_recall,channels_result);
    [actuator_Cim_size, ~] = size(chAMresult);
    
    condition_vectorlist = zeros (channels_condition, D);
    
    i = 1;
    while i <= length_recall
        for x = 1:1:channels_condition
            %[CiM_condition, ~] = lookupItemMemeory (chAMcondition, features_condition(i,x), precision_condition);
            condition_vectorlist(x,:) = xor(chAMcondition(features_condition(i,x)) , iMchcondition(x));
        end
        if channels_condition > 1
            if (mod(channels_condition, 2) == 1)
                condition_vector = mode(condition_vectorlist);
            else
                if (few_all == 0)
                    extra_condition_vector = xor(condition_vectorlist(1,:),condition_vectorlist(2,:));
                else
                    extra_condition_vector = condition_vectorlist(1,:);
                    for y = 2:1:channels_condition
                        extra_condition_vector = xor(extra_condition_vector,condition_vectorlist(y,:));
                    end
                end
                extra_condition_vector = circshift(extra_condition_vector, [1,1]);
                condition_vector = mode([condition_vectorlist; extra_condition_vector]);
            end 
        else
            condition_vector = condition_vectorlist(1,:);
        end
        protected_condition = circshift(condition_vector, [1,1]);    
        
        noisy_resultHV = xor(protected_condition,prog_HV);
        
        %[predict_result, result_error] = hamming(noisy_resultHV, result_AM, result_classes,0); %#ok<ASGLU>
        %result_HV = result_AM(predict_result);
        
        for m = 1:1:channels_result
            noisy_actuator_vector = xor(noisy_resultHV, iMchresult(m));
            [predict_actuator, actuator_error] = hamming(noisy_actuator_vector, chAMresult, actuator_Cim_size,1); %#ok<ASGLU>
            actuator_values(i,m) = predict_actuator;
        end
        i = i + 1;
    end
end
    

function [predict_hamm, error] = hamming (q, aM, classes,iscim)
%
% DESCRIPTION       : computes the Hamming Distance and returns the prediction.
%
% INPUTS:
%   q               : query hypervector
%   AM              : Trained associative memory
%
% OUTPUTS:
%   predict_hamm    : prediction 
%

    sims = [];
    
    if (iscim)
        for j = 0 : classes-1
            sims(j+1) = sum(xor(q,aM(j)));
        end
    else
        for j = 1 : classes
            sims(j) = sum(xor(q,aM(j)));
        end
    end
    
    [error, indx]=min(sims');
    
    if (iscim)
        predict_hamm=indx-1;
    else
        predict_hamm = indx;
    end
     
end
