% start M-LOOP
[status, cmdout] = system('/opt/anaconda3/bin/M-LOOP &');
if status == 0
    disp('M-LOOP successfully launched.');
    fprintf('%s\n', cmdout);
else
    fprintf('Error when launching M-LOOP: \n%s\n', cmdout);
end

while true
    pause(0.5); % check every 0.5s
    if isfile('exp_input.txt')
        % Read
        fid = fopen('exp_input.txt', 'r');
        if fid == -1
            warning('Unable to Open exp_input.txt.');
            continue;
        end
        params = fscanf(fid, '%f');
        fclose(fid);

        delete('exp_input.txt');

        % Run experiment and calculate cost !! to be added !!
        cost = -1; % run_experiment(params);
        uncer = 1;
        bad = 'False';
        
        % Create exp_output_tmp.txt
        fid = fopen('exp_output_tmp.txt', 'w');
        if fid == -1
            warning('Unable to create exp_output_tmp.txt.');
            continue;
        end

        % Write exp_output_tmp.txt
        fprintf(fid, 'cost = %f\n', cost);
        fprintf(fid, 'uncer = %f\n', uncer);
        fprintf(fid, 'bad = %s\n', bad);
        fclose(fid);

        % Move to exp_output.txt
        movefile('exp_output_tmp.txt', 'exp_output.txt');

        % Delete
        % delete('exp_input.txt');
    end
end
