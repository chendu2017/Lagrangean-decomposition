tic
% Maxiteration;  acceptable error; step_size_parameters=alpha; 
% upper bound = inf;  will be updated by langrangean decomposition method
% lower bound = 1  will be unpdated by subgragient optimazition method
Maxiteration = 500;
iteration = 1;
acceptable_error = 0.05;
upper_bound = inf; lower_bound= 1;
step_size = 0.3;
up = zeros(Maxiteration,1);
lp = zeros(Maxiteration,1);
%%%%%module1: generate parameters:
%%(1) the number of advertisements: A
A = 20;
%%(2) the available time slot number: T
T = 40;
%%(3) for each i belongs to A, generate the needed showoff times: Wi
W = unidrnd(20,A,1);   %uniform:1-20
%%(4) for each i belongs to A, generate the benefits si
s = unidrnd(31,A,1)+9;   %uniform:10-40
%%(5) the available size for every time slot: S
S = 50;
%%give the initive langrangean multipier U   
U = rand(A,T);

while ((upper_bound - lower_bound)/lower_bound >= acceptable_error) && (iteration <= Maxiteration)
    %% define decision variables
    X = binvar(A,T);
    V = zeros(A,T);
    sum_1 = 0;
    sum_2 = zeros(1,A);
    %%%%% module2: define&update subproblem
    %% (re-)define subproblem1 for each j. decision variables: Xij   and solve
    for j=1:1:T
        Objective_1 = -(s+U(:,j))'*X(:,j);
        Constraints_1 = [s'*X(:,j) <= S];
        optimize(Constraints_1,Objective_1)
        sum_1 = sum_1 + double(Objective_1);
    end
    X=double(X);
    Objective_1_value = -sum_1;
    %% (re-)define subproblem2 for each i, decision variables: Vij(binary variables)  and solve
    for i=1:1:A
        j=1;
        [B,I] = sort(-U(i,:),'descend');
        while  (j<=T) && (B(1,j)>=0) && (j<=W(i,1)) 
            V(i,I(1,j))=1;
            j=j+1;
        end
        sum_2(1,i)= -U(i,:)*V(i,:)';
    end
    Objective_2_value = sum(sum_2);
    LD_Objective = Objective_1_value + Objective_2_value; 
    up(iteration,1)=LD_Objective;
    V=double(V);
    Objective_2_value = sum(sum_2);
    LD_Objective = Objective_1_value + Objective_2_value;
    
    
    % get objective value from two subproblem, update upper bound
    if LD_Objective <= upper_bound
        upper_bound = LD_Objective;
    end
    
    total_space_i = transpose(sum(V,2).*s);
    %%%%% module4:  do subgradient optimazition by heuristic algorithm based on the solution to subproblem2;
    %%%%%           get the current feasible solution for Primal problem and value then update lower bound
    [~,I]=sort(total_space_i,'descend');    
    occupied_space = zeros(1,T);
    result = zeros(1,T+1);
    for i=1:1:A
        result(I(1,i),1) = I(1,i);
        for t=1:1:T
            if (V(I(1,i),t) == 1) && (V(I(1,i),t)*s(I(1,i),1)+occupied_space(1,t) <= S)
                result(I(1,i),t+1) = V(I(1,i),t);
                occupied_space(1,t) = occupied_space(1,t) + V(I(1,i),t)*s(I(1,i),1);
            end
        end
    end
    %% lower bound
    P_Objective = sum(result(:,2:T+1),2)'*s(result(:,1),1);
    lp(iteration,1)= P_Objective;
    if P_Objective >= lower_bound
        lower_bound = P_Objective;
        best_solution = result;
    end
    
    %%%%% module5:
    %% update U
    U = U + (step_size*(P_Objective-LD_Objective)*(X - V))/sumsqr(X-V);
    iteration = iteration + 1;
end
gap = (upper_bound - lower_bound)/lower_bound*100;
toc 