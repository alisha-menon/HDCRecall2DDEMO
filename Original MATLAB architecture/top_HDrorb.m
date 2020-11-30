clear all
%HD recall of reactive behavior
%data format - columns for each feature/channel of data 
%====Features and Label===
%% select = 0 for regular, select = 1 for known condition tracking in training
select = 0;
% few_all = 0 for the extra bundling vector to be a permuted version of the
% xor of 2 features (better for larger number of channels)
% few_all = 1 for the extra bundling vector to be a permuted version of the
% xor of all the features (better for smaller number of channels)
few_all = 1;
%% set these:
%maxL_condition_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]; % # of vectors in CiM
%maxL_result_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]; % # of vectors in CiM
maxL_condition_list = [2]; % # of vectors in CiM
maxL_result_list = [4]; % # of vectors in CiM
%count_list = [20, 40, 80, 120, 160, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
%maxL_condition_list = 20;
%maxL_result_list = 20;
num_channels = 64;
%num_result_channels_list = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100];
num_result_channels_list = [1];
num_condition_channels_list = [4];
count_list = (2^num_condition_channels_list)*4;
%num_condition_channels_list = [20, 40, 60];
%num_condition_channels_list = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100];
D = 10000; %dimension of the hypervectors
features_result = [];
features_condition = [];
learningrate=0.25; % percentage of the dataset used to train the algorithm
accuracy_list = zeros(length(count_list), min(length(maxL_condition_list), length(maxL_result_list)));
error_due_to_repeat = zeros(length(count_list), min(length(maxL_condition_list), length(maxL_result_list)));

for g = 1:length(num_result_channels_list)
         num_result_channels = num_result_channels_list(g);
         num_condition_channels = num_condition_channels_list(g);
         count = count_list;
    %% works
%     count = count/10;
%      features_result = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%      features_condition = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%      for i = 2:1:count
%          features_result = [features_result; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%      end
%      features_result_1 = features_result;
%      for i = 2:1:num_result_channels
%          features_result = [features_result features_result_1];
%      end
%      for i = 2:1:count
%          features_condition = [features_condition; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%      end
%      features_condition_1 = features_condition;
%      for i = 2:1:num_condition_channels
%          features_condition = [features_condition features_condition_1];
%      end

    %% randomly select pairs for training and then select from previous dataset for recall
    features_condition = (dec2bin(0:2^num_condition_channels-1)' - '0')';
    for i = 1:length(features_condition)
        choices = (find(features_condition(i,:) == 0))-1;
        if size(choices,2) > 1
            randselected_result = randsample(choices,1);
        elseif size(choices,2) == 1
            randselected_result = choices;
        else
            randselected_result = 0; 
        end
        features_result = [features_result; randselected_result]; %#ok<AGROW>
    end
    %features_result = [features_result; features_result]; %#ok<AGROW>
    %features_condition = [features_condition; features_condition]; %#ok<AGROW>
    
    % multiple channels randomly select pairs for training and then select from previous dataset for recall
    for i = 1:1:(count - count*learningrate)
        x = randi([1 count*learningrate],1,1);
        pair_result = features_result(x,:);
        pair_condition = features_condition(x,:);
        features_result = [features_result ; pair_result]; %#ok<AGROW>
        features_condition = [features_condition ; pair_condition]; %#ok<AGROW>
    end

    %% =======HDC============
    if (select == 1)
            HD_functions_HDrorb_knowncondition;     % load HD functions with known condition clause
    else
        HD_functions_HDrorb;     % load HD functions
    end
  for j=1:length(maxL_condition_list)
    learningFrac = learningrate; 
    maxL_condition = maxL_condition_list(j);
    maxL_result = maxL_result_list(j);
    %% generate random HVs for conditions and result item memories and ciM
    [chAMcondition, iMchcondition, chAMresult, iMchresult] = initItemMemories (D, maxL_condition, num_condition_channels, maxL_result, num_result_channels);
    
    %% designate training vs. recall data
    features_condition_training = features_condition(1:floor(count*learningrate),:); %for now..
    features_result_training = features_result(1:floor(count*learningrate),:); %for now..
    features_condition_recall = features_condition(floor(count*learningrate)+1:count,:); %for now..
    features_result_recall = features_result(floor(count*learningrate)+1:count,:); %for now..
    length_training = size(features_condition_training,1);
    length_recall = size(features_condition_recall,1);

    fprintf ('HDC for RORB\n');
    if (select == 1)
        [prog_HV, prog_HVlist, result_AM, known_condition_integers, result_integers] = hdcrorbtrain (few_all, length_training, features_condition_training, features_result_training, chAMcondition, chAMresult, iMchcondition, iMchresult, D, precision_condition, precision_result, num_condition_channels, num_result_channels); 
    else
        [prog_HV] = hdcrorbtrain (few_all, length_training, features_condition_training, features_result_training, chAMcondition, chAMresult, iMchcondition, iMchresult, D, num_condition_channels, num_result_channels); 
    end
    
    [actuator_values] = hdcrorbpredict (few_all, length_recall, prog_HV, features_condition_recall, chAMcondition, chAMresult, iMchcondition, iMchresult, D, num_condition_channels, num_result_channels); 
    expected = int64 (features_result_recall);

    %% check the number of repeat condition samples
%     non_repeat_errors = [];
     row_wrong = [];
     for i = 1:1:length_recall
        if (sum(expected(i,:) == actuator_values(i,:)) < num_result_channels)
            row_wrong = [row_wrong i];
        end
     end
     [A,u,c] = unique(features_condition_training,'rows');
     [n,~,~] = histcounts(c,numel(u));
    repeat_bin_address = find(n > 1);
    repeat_address_condition = u(repeat_bin_address);
    repeat_values_condition = features_condition_training(repeat_address_condition,:);
    [length_repeat, ~] = size(repeat_values_condition);
    wrong_recall_values_condition = features_condition_recall(row_wrong,:);
    match_count_condition = 0;
    [length_wrong, ~] = size(wrong_recall_values_condition);
    non_repeat_errors = [];
    for i = 1:1:length_wrong
        x = 0;
        for o = 1:1:length_repeat
            if (repeat_values_condition(o,:) == wrong_recall_values_condition(i,:))
                x = 1;
            end
        end
        if (x == 1)
            match_count_condition = match_count_condition+1;
        else
            non_repeat_errors = [non_repeat_errors; wrong_recall_values_condition(i,:)]; %#ok<AGROW>
        end
    end
    [~,u,c] = unique(features_result_training,'rows');
    [n,~,bin] = histcounts(c,numel(u));
    repeat_bin_address = find(n > 1);
    repeat_address_result = u(repeat_bin_address);
    repeat_values_result = features_result_training(repeat_address_result,:);
    [length_repeat_result, ~] = size(repeat_values_result);
    match_count_result = 0;
    wrong_recall_values_result = features_result_recall(row_wrong,:);
    [length_wrong_result, ~] = size(wrong_recall_values_result);
    for i = 1:1:length_wrong_result
        for o = 1:1:length_repeat_result
            x = x+(repeat_values_result(o,:) == wrong_recall_values_result(i,:));
        end
        if (x > 0)
            match_count_result = match_count_result+1;
        end
    end

    %% check whether the result samples were the same for all of the repeat condition samples
    result_repeat_count = zeros(length_repeat,4);
    location = [];
    for i = 1:1:length_repeat
        for o = 1:1:length_training
            x = (repeat_values_condition(i,:) == features_condition_training(o,:));
            if (x == 1)
                location = [location o];
            end
        end
        %location = find(quantized_condition_training == repeat_values_condition(i));
        unique_values = unique(features_result_training(location),'rows');
        if (length(unique_values) == length(location))
            result_repeat_count(i,1) = length(location);
            result_repeat_count(i,2) = 0;
        else
            result_repeat_count(i,1) = length(location);
            result_repeat_count(i,2) = length(location) - length(unique_values);
        end
        result_repeat_count(i,3) = result_repeat_count(i,1) - result_repeat_count(i,2);
        if (result_repeat_count(i,3) > 0)
            result_repeat_count(i,4) = 0;
        else

            result_repeat_count(i,4) = 1;
        end
    end
    number_unique_condition = sum(result_repeat_count(:,4));
    training_sample_count = count*learningrate;
    recall_sample_count = count*learningrate;

    training_sample_count
    num_condition_channels
    num_result_channels
    recall_sample_count; 
    error_due_to_repeat(g, j) = length(row_wrong) == match_count_condition;

    %% Check accuracy for eevery actuator value 
    maxL_condition
    maxL_result;
    [expected_length, ~] = size(expected);
    accuracy_count = 0;
    for h = 1:1:expected_length
        channel_match_count = 0;
        for u = 1:1:num_result_channels
            match = (expected(h,u) == actuator_values(h,u));
            channel_match_count = channel_match_count + match;
        end
        if channel_match_count == num_result_channels
            accuracy_count = accuracy_count + 1;
        end
    end
    %% Final accuracy and print
    % accuracy_2 is do all channels match for each sample
    accuracy_2 = accuracy_count/expected_length*100
    % accuracy is the overall accuracy of each individual actuator value
    accuracy = sum(sum(actuator_values==expected))/numel(expected)*100
    next_to = [expected actuator_values];
    accuracy_list(g, j) = accuracy;
    accuracy_list_2(g, j) = accuracy_2;
    [length_row_wrong, ~] = size(row_wrong);
    %[length_quantized_result_recall, ~] = size(quantized_result_recall);
    %fprintf('%d of the %d recall samples were wrong, %d of these matched the repeats in the condition samples where %d of the repeated condition values also had the same result value for all the samples, and %d of these matched the repeats in the result samples\n',length(row_wrong), length(quantized_result_recall), match_count_condition, number_unique_condition, match_count_result);
    fprintf('%d of the %d recall samples were wrong, %d of these matched the repeats in the condition samples where %d of the repeated condition values also had the same result sample for all the samples\n',length(row_wrong), length_recall, match_count_condition, number_unique_condition);

    %accuracy(N,2) = acc1;
    %acc1

    %acc_ngram_1(N,j)=acc1;
    %acc_ngram_1(N,j)=acc1;
    end
end

    
