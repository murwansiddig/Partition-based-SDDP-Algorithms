
#Time limit function 
#using Distributed
@everywhere function run_with_timeout(timeout::Int,f::Function, wid::Int)
    result = RemoteChannel(()->Channel{Tuple}(1));
    @spawnat wid put!(result, (f(),myid()))
    res = (:timeout, wid)
    time_elapsed = 0.0
    hr = 3600;
    while time_elapsed < timeout && !isready(result)
        sleep(0.5)
        time_elapsed += 0.5
        if time_elapsed == hr 
            println("Time = ", time_elapsed);
            hr += 3600; 
        end
    end
    if !isready(result)
        println("Timeout! at $wid")
    else
        res = take!(result)
    end
    return res
end

#Function for the special case when we want to enumerate all possible Sample Paths
@everywhere struct WithRepetitionsPermutations{T}
    a::T
    t::Int
end
 
@everywhere with_repetitions_permutations(elements::T, len::Integer) where T =
    WithRepetitionsPermutations{T}(unique(elements), len)
 
Base.iteratorsize(::WithRepetitionsPermutations) = Base.HasLength()
Base.length(p::WithRepetitionsPermutations) = length(p.a) ^ p.t
Base.iteratoreltype(::WithRepetitionsPermutations) = Base.HasEltype()
Base.eltype(::WithRepetitionsPermutations{T}) where T = T
Base.start(p::WithRepetitionsPermutations) = ones(Int, p.t)
Base.done(p::WithRepetitionsPermutations, s::Vector{Int}) = s[end] > endof(p.a)
function Base.next(p::WithRepetitionsPermutations, s::Vector{Int})
    cur = p.a[s]
    s[1] += 1
    local i = 1
    while i < endof(s) && s[i] > length(p.a)
        s[i] = 1
        s[i+1] += 1
        i += 1
    end
    return cur, s
end

#function that generate coarse cuts give a partion P and time period t
@everywhere function SPC(data,MasterSubData,xxval,P,t)
    tt=t+1

    piC = zeros(length(P),nbHydroP);
    QC = zeros(length(P));
    
    #For each cluster we will do the following
    for k=1:length(P)
        for h in HReserv
            JuMP.setRHS(
            FBconstr[tt][h],
                
            xxval[h,tt-1]+
            c0*(sum(scen[1+nbRealization*(tt-2)+j,2+h] for j in P[k])/length(P[k]))
                       );
        end
        
        for h in HNoReserv
            JuMP.setRHS(
            FBconstr[tt][h],
            
            sum(scen[1+nbRealization*(tt-2)+j,2+h] for j in P[k])/length(P[k])
                       );
        end
        
        statusSPC = solve(MasterSub[tt]);
        QC[k] = getobjectivevalue(MasterSub[tt]);

        for h=1:nbHydroP
            piC[k,h] = getdual(FBconstr[tt][h]);
        end
    end
    return QC, piC
end

#function that generate semi-coarse cuts give a partion P and time period t
@everywhere function SPSC(data,MasterSubData,xxval,P,c,t,QSC,piSC)
    tt=t+1

    #For each scenario in the cluster c we will do the following 
    for k in P[c]
        for h in HReserv
            JuMP.setRHS(
            FBconstr[tt][h],
                
            xxval[h,tt-1]+
            c0*(scen[1+nbRealization*(tt-2)+k,2+h])
                       );
        end
        
        for h in HNoReserv
            JuMP.setRHS(
            FBconstr[tt][h],
                
            scen[1+nbRealization*(tt-2)+k,2+h]
                       );
        end
        
        #We solve the subproblen & obtain its objective value
        statusSPSC = solve(MasterSub[tt]);
        QSC[k] = getobjectivevalue(MasterSub[tt]);
        
        #We get the dual multiplier of each constraint for this realization
        for h=1:nbHydroP
            piSC[k,h] = getdual(FBconstr[tt][h]);
        end
    end
    
    return QSC, piSC 
end


@everywhere function Refine(P,PC,g,piSC,eps,QSC,t,xxval)
    temp = [];
    reps = [];
    push!(temp,[PC[g][1]])
    push!(reps,piSC[PC[g][1],:]);
    
    for k=2:length(PC[g])
        count = 0;
        for s=1:length(temp)
            diff = piSC[PC[g][k],:]-reps[s];
            if norm(diff) < eps
                push!(temp[s],PC[g][k])
                break
            else
                #check if the solution is degenerate solutions
                obj_diff = sum(piSC[PC[g][k],h]*xxval[h,t] for h in HReserv) - sum(reps[s][h]*xxval[h,t] for h in HReserv)
                    
                if abs(obj_diff)/sum(reps[s][h]*xxval[h,t] for h in HReserv) < eps
                    push!(temp[s],PC[g][k])
                    break
                else
                    count +=1
                end
            end
        end
        if count == length(temp)
            push!(temp,[PC[g][k]])
            push!(reps, piSC[PC[g][k],:]);
        end
    end 
    
    filter!(e->e!=PC[g],P)
    for i=1:length(temp)
        push!(P,temp[i])
    end

    return P
end
@everywhere function Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k)
    #Type_1 cut == coarse cut 
    P = Part[t+1]
    cut_added = 0;
      
    QC, piC = SPC(data,MasterSubData,xxval,P,t);
    
    θ = sum(QC[j]*(length(P[j])/nbRealization) for j=1:length(P));
    
    abs_gap = (θ-tthetaval[t]);
    if abs_gap < 1e-5
        eps = 1e-1
    end
    if abs_gap/max(1e-8,abs(tthetaval[t])) > eps && abs_gap > 1e-6
        ##########################################################
        @constraint(
        MasterSub[t], 
        
        thetavar[t]-
        sum(sum(piC[j,h]*xvar[t][h,t] for h in HReserv)*(length(P[j])/nbRealization) for j=1:length(P))
        >=
        θ-
        sum(sum(piC[j,h]*xxval[h,t] for h in HReserv)*(length(P[j])/nbRealization) for j=1:length(P))
            
                   );
        
        cut_added = 1;
        ##########################################################
    end
    return cut_added, QC, piC
end

@everywhere function Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC)
    #Type_2 cut == semi-coarse cut
    cut_counter = 0;
    #sort!(Part[t+1], by=length, rev=true)
    P = Part[t+1]
    p= length(P);
    PC = deepcopy(P)
    
    piSC = zeros(nbRealization,nbHydroP);
    QSC = zeros(nbRealization);
    
    for c=1:length(PC)
        #solve scenario-subproblem for every realization in the cluster
        
        
        if  c != length(PC) && length(PC[c]) > 1 
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            QSC, piSC = SPSC(data,MasterSubData,xxval,P,c,t,QSC,piSC);
            θ = sum(sum(QSC[k] for k in PC[g])*(1/nbRealization) for g=1:c)+
                sum(QC[k]*(length(PC[k])/nbRealization) for k=c+1:length(PC))
            
            abs_gap = (θ-tthetaval[t]);
            if abs_gap < 1e-5
                eps = 1e-1
            end
            if abs_gap/max(1e-8,abs(tthetaval[t])) > eps && abs_gap > 1e-6
            ##############################################################################################
                @constraint(
                MasterSub[t],
                    
                thetavar[t]-(
                sum(sum(sum(piSC[j,h]*xvar[t][h,t] for h in HReserv)*(1/nbRealization) for j in PC[g]) for g=1:c)+
                sum(sum(piC[j,h]*xvar[t][h,t] for h in HReserv)*(length(PC[j])/nbRealization) for j=c+1:length(PC))
                            )
                >=
                θ-
                sum(sum(sum(piSC[j,h]*xxval[h,t] for h in HReserv)*(1/nbRealization) for j in PC[g]) for g=1:c)-
                sum(sum(piC[j,h]*xxval[h,t] for h in HReserv)*(length(PC[j])/nbRealization) for j=c+1:length(PC))
                           );

                 cut_counter += 1
                 for g=1:c
                     #P = Refine(P,PC,g,piSC,eps)
                     P = Refine(P,PC,g,piSC,eps,QSC,t,xxval)
                 end
                
                 break
            ##############################################################################################
            end
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        elseif c == length(PC) && length(PC[c]) > 1
            QSC, piSC = SPSC(data,MasterSubData,xxval,P,c,t,QSC,piSC);
            θ = sum(QSC[k] for k=1:nbRealization)*(1/nbRealization);
            abs_gap = (θ-tthetaval[t]);
            if abs_gap < 1e-5
                eps = 1e-1
            end
            if abs_gap/max(1e-8,abs(tthetaval[t])) > eps && abs_gap > 1e-6
            ########################################################
                
                @constraint(
                MasterSub[t],
                    
                thetavar[t]-
                sum(sum(sum(piSC[j,h]*xvar[t][h,t] for h in HReserv)*(1/nbRealization) for j in PC[g]) for g=1:c)
                >=
                θ-
                sum(sum(sum(piSC[j,h]*xxval[h,t] for h in HReserv)*(1/nbRealization) for j in PC[g]) for g=1:c)
                           );
                
                cut_counter += 1;
                
                for g=1:c
                    #P = Refine(P,PC,g,piSC,eps)
                    P = Refine(P,PC,g,piSC,eps,QSC,t,xxval)
                end
            ########################################################    
            end
        end
    end

    return P, cut_counter
end

@everywhere function Type3_cut(MasterSubData,data,t,xval,thetaval,eps,SP,k)
    
    t +=1
    cut_count = 0
    SPobj = zeros(nbRealization)
    SPpi = zeros(nbHydroP,nbRealization)

    for j=1:nbRealization
        if t>1
            for h in HReserv
                JuMP.setRHS(FBconstr[t][h], xval[h,k,t-1]+c0*scen[nbRealization*(t-2)+1+j,2+h]);
            end
            for h in HNoReserv
                JuMP.setRHS(FBconstr[t][h], scen[nbRealization*(t-2)+1+j,2+h]);
            end                     
        end
        status = solve(MasterSub[t]);
        SPobj[j]= getobjectivevalue(MasterSub[t]);
        for h=1:nbHydroP
            SPpi[h,j]= getdual(FBconstr[t][h]);
        end
    end
    θ = sum(SPobj[j] for j=1:nbRealization)*(1/nbRealization);
    
    abs_gap = (θ-thetaval[k,t-1]);
    if abs_gap < 1e-5
        eps = 1e-1
    end
    if abs_gap/max(1e-8,abs(thetaval[k,t-1])) > eps && abs_gap > 1e-5
        @constraint(
        MasterSub[t-1],
        thetavar[t-1]-
        (1/nbRealization)*sum(sum(SPpi[h,j]*xvar[t-1][h,t-1] for h in HReserv) for j=1:nbRealization)
        >=
        θ-
        (1/nbRealization)*sum(sum(SPpi[h,j]*xval[h,k,t-1] for h in HReserv) for j=1:nbRealization));
        cut_count +=1
    end

    
    return cut_count
end

@everywhere function Type4_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k)
    #Type_2 cut == semi-coarse cut
    cut_counter = 0;
    #sort!(Part[t+1], by=length, rev=true)
    P = Part[t+1]
    p= length(P);
    PC = deepcopy(P)
    
    piSC = zeros(nbRealization,nbHydroP);
    QSC = zeros(nbRealization);
    θ = 0
    for c=1:length(PC)
        #solve scenario-subproblem for every realization in the cluster
        QSC, piSC = SPSC(data,MasterSubData,xxval,P,c,t,QSC,piSC);
        θ += sum(QSC[k] for k in PC[c])*(1/nbRealization)
    end
    
    abs_gap = (θ-tthetaval[t]);
    if abs_gap < 1e-5
        eps = 1e-1
    end
    if abs_gap/max(1e-8,abs(tthetaval[t])) > eps && abs_gap > 1e-6
    ##############################################################################################
        @constraint(
        MasterSub[t],

        thetavar[t]-(
        sum(sum(sum(piSC[j,h]*xvar[t][h,t] for h in HReserv)*(1/nbRealization) for j in PC[c]) for c=1:length(PC))
                    )
        >=
        θ-(
        sum(sum(sum(piSC[j,h]*xxval[h,t] for h in HReserv)*(1/nbRealization) for j in PC[c]) for c=1:length(PC))
          )
                   );

         cut_counter += 1
         for c=1:length(PC)
             #P = Refine(P,PC,c,piSC,eps)
             P = Refine(P,PC,g,piSC,eps,QSC,t,xxval)
         end

    ##############################################################################################
    end

    return P, cut_counter
end

@everywhere function forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB)

    UB_SP = zeros(nbSP);
    for k=1:nbSP
        tempUB = 0;
        for t=1:nbStages            
            J=SP[k,t]
            j= convert(Int64, J)
            #We start updating the RHS of flow balance constraint
            if t>1
                for h in HReserv
                    JuMP.setRHS(FBconstr[t][h], xval[h,k,t-1]+c0*scen[nbRealization*(t-2)+1+j,2+h]);
                end
                for h in HNoReserv
                    JuMP.setRHS(FBconstr[t][h], scen[nbRealization*(t-2)+1+j,2+h]);
                end                                
            end
            status = solve(MasterSub[t]);
            if t==1
                LB = getobjectivevalue(MasterSub[t]);
            end
            for h=1:nbHydroP
                xval[h,k,t]= getvalue(xvar[t][h,t]);
            end
            if t<nbStages
                thetaval[k,t] = getvalue(thetavar[t]);
            end
            local_cost = sum(cf[f]*getvalue(gvar[t][f,t]) for f=1:nbThermoP)+cp*getvalue(pvar[t]);
            tempUB +=local_cost
        end
        UB_SP[k]=tempUB
    end
    UB = sum(UB_SP[k] for k=1:nbSP)/nbSP
    
    return LB, xval, thetaval, UB
end

@everywhere function inner_forward_pass(MasterSubData,data,nbSP,SP,xxval,tthetaval,range,k)
    filter!(a->a>=1, range)

    for t=range[1]:range[end]            
        J=SP[k,t]
        j= convert(Int64, J)
        #We start updating the RHS of flow balance constraint
        if t>1
            for h in HReserv
                JuMP.setRHS(FBconstr[t][h], xxval[h,t-1]+c0*scen[nbRealization*(t-2)+1+j,2+h]);
            end
            for h in HNoReserv
                JuMP.setRHS(FBconstr[t][h], scen[nbRealization*(t-2)+1+j,2+h]);
            end                                
        end
        status = solve(MasterSub[t]);
        if t==1
            LB = getobjectivevalue(MasterSub[t]);
        end
        for h=1:nbHydroP
            xxval[h,t]= getvalue(xvar[t][h,t]);
        end
        if t<nbStages
            tthetaval[t] = getvalue(thetavar[t]);
        end
    end

    
    return xxval, tthetaval
end

@everywhere function seasons_class(MasterSubData,data,PARAMS)
    if PARAMS[2] == "D"
        threshold = 8000
    else
        threshold = 500000
    end
    cost = zeros(nbStages)
    All_cost = []
    means_cost = [];
    Wet = [];
    Dry = [];
    for t=2:nbStages
        xval = zeros(nbHydroP);
        local_cost = zeros(nbRealization)
        for j=1:nbRealization
            #We start updating the RHS of flow balance constraint
            if t>1
                for h in HReserv
                    JuMP.setRHS(FBconstr[t][h], xval[h]+c0*scen[nbRealization*(t-2)+1+j,2+h]);
                end
                for h in HNoReserv
                    JuMP.setRHS(FBconstr[t][h], scen[nbRealization*(t-2)+1+j,2+h]);
                end                                
            end
            status = solve(MasterSub[t]);
            local_cost[j] = getobjectivevalue(MasterSub[t]);
            push!(All_cost, local_cost[j])
        end

        push!(means_cost,mean(local_cost))
        if mean(local_cost) >= threshold
            push!(Dry,t)
        else
            push!(Wet,t)
        end
    end
    Wet_seasons = [];
    Dry_seasons = [];
    t = 1;
    while true
        temp = [];
        if t == length(Dry)
            break
        end
        while true
            if t < length(Dry)
                if Dry[t+1]-Dry[t] <=1
                    push!(temp,Dry[t])
                    t += 1
                else
                    push!(temp,Dry[t])
                    t +=1
                    break
                end

            else
                push!(temp,Dry[t])
                break
            end 
        end
        push!(Dry_seasons,temp)
    end

    t = 1
    while true
        temp = [];
        while true
            if t < length(Wet)
                if Wet[t+1]-Wet[t] <=1
                    push!(temp,Wet[t])
                    t += 1
                else
                    push!(temp,Wet[t])
                    t +=1
                    break
                end
            else
                push!(temp,Wet[t])
                break
            end 
        end
        push!(Wet_seasons,temp)
        if t == length(Wet)
            break
        end
    end


    Seasons = [];
    Seasons = vcat(Seasons,Dry_seasons)
    Seasons = vcat(Seasons,Wet_seasons)
    sort!(Seasons, by = x -> x[1]);
    Wet_index = findin(Seasons,Wet_seasons);
    Dry_index = findin(Seasons,Dry_seasons);
    
    println("Wet = ", Wet)
    println("Dry = ", Dry)
    println("means_cost =", means_cost)
    println(" =================================================")
    println(" =================================================")
    
    return Seasons, Wet_index, Dry_index, Wet, Dry
end

#Function to create the coarse_tree based on the new partion 
@everywhere function NewScen(Part,scen,nbStages,nbRealization,nbHydroP)
    newscen = zeros(sum(length(Part[t]) for t=1:nbStages),length(Scenarios[1]));
    newscen[1,:] = scen[1,:];
    
    row_count = 1
    for t=2:nbStages
        for c=1:length(Part[t])
            row_count +=1
            newscen[row_count,1]=t
            newscen[row_count,2]=c
            for h=1:nbHydroP
                newscen[row_count,2+h] = sum(scen[nbRealization*(t-2)+1+j,2+h] for j in Part[t][c])/length(Part[t][c])
            end
        end
    end
    return newscen
end

#Function to enumrate all possible Sample path if the number of realaization is different between different stages 

@everywhere function AllSP(Part,nbStages,nbRealization)
    
    nbSP = prod(length(Part[t]) for t=1:nbStages);
    SP = zeros(nbSP,nbStages);
    SP_prob = zeros(nbSP);
    
    rowcount =0
    main = ones(nbStages)
    while true 
        rowcount += 1
        SP[rowcount,:] =main
        for t=nbStages:-1:2
            if main[t] < length(Part[t])
                main[t] +=1
                break
            else main[t]==length(Part[t])
                if main[t-1] < length(Part[t-1])
                    main[t-1]+=1
                    for tt=t:nbStages
                        main[tt]=1
                    end
                    break
                else
                    continue
                end
            end
        end
        if rowcount==nbSP
            break;
        end
    end  
    
    for s=1:nbSP
        prob = 1;
        for t=2:nbStages
            J=SP[s,t]
            j= convert(Int64, J)
            if length(Part[t])>1
                prob *= length(Part[t][j])/nbRealization
            else
                prob *= 1
            end
        end
        if prob < 0 
            SP_prob[s] = 0
        else
            SP_prob[s] = prob
        end
    end   
    
    return SP, SP_prob
end

#Function to generate a random number with a non-uniform probability
@everywhere function RandProb(Part,nbStages,nbRealization,nbSP)
    
    p = [];
    push!(p,[1])
    for t=2:nbStages
        temp = Array{Float64,1}(length(Part[t]));
        for c=1:length(Part[t])
            temp[c] = length(Part[t][c])./nbRealization
        end
        push!(p,temp)
    end
    
    SP = fill(1, (nbSP,nbStages))
    SP_prob = fill(1.0, (nbSP))
    SP[:,1]=1;
    for s=1:nbSP
        prob = 1
        for t=2:nbStages
            if length(p[t]) > 1
                d = Multinomial(1, p[t])
                SP[s,t] = findin(rand(d,1),1)[1]
            else
                SP[s,t] = 1
            end
            prob *= p[t][SP[s,t]]
        end
        SP_prob[s] = prob
    end
    
    return SP, SP_prob
end

#Initialization

@everywhere function initialize(nbRealization,nbStages,nbHydroP);

    ###############################################################################
    ###############################################################################
    indicator = 0;
    if nbRealization^(nbStages-1) < 1000 && nbRealization^(nbStages-1) > 0
        nbSP = nbRealization^(nbStages-1);
        SP = zeros(nbSP,nbStages);
        SP[:,1]=1;
        Realization_list = [];
        for t=1:nbRealization
            push!(Realization_list,t)
        end
        templist = collect(with_repetitions_permutations(Realization_list, nbStages-1));
        for k=1:nbSP
            for t=1:nbStages-1
                SP[k,t+1]=templist[k][t]
            end
        end
    else
        indicator =1;
        nbSP = 1;
        SP = zeros(nbSP,nbStages);
        SP[:,:]=rand(1:nbRealization, nbSP,nbStages);
        SP[:,1]=1;
    end
    ###############################################################################
    ###############################################################################
    #Initialize a Matrix for the objective values of Master problem: per Sample Path, Per Stage 
    objval = zeros(nbSP,nbStages);

    #Initialize a Matrix for the xvalues: per Sample Path, Per Stage (xval gets updated only when we are solving the forward pass)
    xval = zeros(nbHydroP,nbSP,nbStages);

    #Initialize a Matrix for the thetavalues: per Sample Path, Per Stage 
    thetaval = zeros(nbSP,nbStages);

    #Initialize a Vector for the Upper bound per every Sample Path 
    UB_SP = zeros(nbSP);

    optimalx_1 = zeros(nbHydroP);
    ETGF = 0;

    LB = MVPobj[1];
    Local_UB = 1e10;
    UB = 1e10;
    iter = 0;
    eps = 1e-4
    
    LBs = [];
    TIMEOUT = 10800;
    
    return nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator
    
end

#initial partition, all scenarios aggregated together 
@everywhere function initial_Partition(nbRealization,nbStages); 
    IP = [];
    for t=1:nbStages
        if t==1
            list = [1]
            push!(IP,list)
        else
            templist1 = [];
            templist2 = [];
            for k=1:nbRealization
                push!(templist2,k)
            end
            push!(templist1,templist2);
            push!(IP,templist1)
        end
    end
    return IP
end


#coarse tree forward pass
@everywhere function coarse_tree_FP(MasterSubData,data,nbSP,SP,xval,thetaval,LB,Part,SP_prob)
    UB_SP = zeros(nbSP);
    for k=1:nbSP
        tempUB = 0;
        for t=1:nbStages            
            J=SP[k,t]
            j= convert(Int64, J)
            #We start updating the RHS of flow balance constraint
            if t>1
                
                for h in HReserv
                    RHS = sum(scen[nbRealization*(t-2)+1+k,2+h] for k in Part[t][j])/length(Part[t][j]);
                    JuMP.setRHS(FBconstr[t][h], xval[h,k,t-1]+c0*RHS);
                end
                for h in HNoReserv
                    RHS = sum(scen[nbRealization*(t-2)+1+k,2+h] for k in Part[t][j])/length(Part[t][j]);
                    JuMP.setRHS(FBconstr[t][h], RHS);
                end                                
            end
            status = solve(MasterSub[t]);
            if t==1
                LB = getobjectivevalue(MasterSub[t]);
            end
            for h=1:nbHydroP
                xval[h,k,t]= getvalue(xvar[t][h,t]);
            end
            if t<nbStages
                thetaval[k,t] = getvalue(thetavar[t]);
            end
            local_cost = sum(cf[f]*getvalue(gvar[t][f,t]) for f=1:nbThermoP)+cp*getvalue(pvar[t]);
            tempUB +=local_cost
        end
        UB_SP[k]=tempUB
    end
    UB = sum(UB_SP[k]*SP_prob[k] for k=1:nbSP)
    
    return LB, xval, thetaval, UB
end
