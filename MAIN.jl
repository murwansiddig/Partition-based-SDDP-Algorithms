
PARAMS = ARGS;
TIMEOUT = 60*60*48;
#time_limit = 60*60*6;
srand(2019);


wid = addprocs(1)[1];
############################################################
include("packages.jl");
include("problem_data.jl");
include("model.jl");
include("auxiliary_functions.jl");
############################################################


function ff()
    iter = 0;
    ALL_LBS = [];
    TIMES = [];
    Refinement_time = [];
    LB_per_hr = [];
    UB_per_hr = [];
    CI_per_hr = [];
    ALL_cuts_counter = fill(0,3,nbStages);
    start=time();
    println("start_time = ", start)
    nbSP_F = 1;
    nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
    initialize(nbRealization,nbStages,nbHydroP,nbSP_F);   
    
    PartsCount = ones(nbStages); #array to keep track the size of partition at every time period 
    PartsCountC = ones(nbStages);
    Part = initial_Partition(nbRealization,nbStages); 
    
    if PARAMS[1] == "VLVA_QP"
        Seasons, Wet_index, Dry_index, Wet, Dry = seasons_class(MasterSubData,data,PARAMS);
    end
    
    if PARAMS[1] == "PrePro" || PARAMS[1] == "Tree_Parts"
        phase3 = 0.5 
        TotReal = sum(PartsCount[t]/nbRealization for t=2:nbStages)/(nbStages-1);
    end
    
    ub_calc_time = 0;
    #for time_limit in [60*60*1/60, 60*60*2/60, 60*60*3/60]
    #for time_limit in [60*60*1, 60*60*3, 60*60*6, 60^2*12, 60^2*24, 60^2*48]
    for time_limit in [0, 1, 2, 3, 4, 5]

    #for time_limit in [60*60*0.05, 60*60*0.1]
        nbSP_F = 1;
        nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
        
        if PARAMS[1] == "SDDP_QP" || PARAMS[1] == "SDDP_CP"
            while true
                iter += 1;
                LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                println("iter = ", iter);
                println("LB = ", LB);
                println("UB = ", UB);
                Elapsed = time() - start - ub_calc_time;
                push!(ALL_LBS,LB)
                push!(TIMES,Elapsed)

                if Elapsed>time_limit
                    CutCount_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_",time_limit,"_CC.csv");
                    CC_df = DataFrame(ALL_cuts_counter);
                    #CSV.write(CutCount_file_name,CC_df);
                    ub_calc_stime = time();
                    nbSP_F = 10000;
                    nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                    LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                    μ = mean(UB_SP);
                    ci = 1.96*std(UB_SP)/sqrt(nbSP);
                    push!(LB_per_hr,LB)
                    push!(UB_per_hr,μ)
                    push!(CI_per_hr,ci)
                    println("time passed = ", Elapsed/(60^2))
                    println("Lower bound: ", LB);
                    println("Confidence interval: ", μ, " ± ", ci)
                    ub_calc_time = time() - ub_calc_stime
                    break
                end
                println("========================================================================");
                println("========================================================================");
                println(" ");

                if PARAMS[1] == "SDDP_QP"
                    #<<<<<<<<<<SDDP_QP>>>>>>>>>>>>>>>
                    ###################################################################################
                    ###################################################################################
                    for k=1:nbSP
                        for t=nbStages-1:-1:1
                            cut_count = Type3_cut(MasterSubData,data,t,xval,thetaval,eps,SP,k);
                            ALL_cuts_counter[3,t]+= cut_count;
                        end
                    end 
                else PARAMS[1] == "SDDP_CP"
                    #<<<<<<<<<<SDDP_CP>>>>>>>>>>>>>>>
                    ###################################################################################
                    ###################################################################################
                    for k=1:nbSP
                        xxval = xval[:,k,:]
                        tthetaval = thetaval[k,:]
                        for t=nbStages-1:-1:1
                            while true
                                cut_count = 0;
                                cut_count += Type3_cut(MasterSubData,data,t,xval,thetaval,eps,SP,k);
                                ALL_cuts_counter[3,t]+= cut_count;
                                if cut_count == 0 
                                    break;
                                else            
                                    range = [t];
                                    xxval, tthetaval = inner_forward_pass(MasterSubData,data,nbSP,SP,xxval,tthetaval,range,k);
                                    xval[:,k,:] = xxval;
                                    thetaval[k,:] = tthetaval;
                                end
                            end
                        end
                    end
                end

                if indicator == 1
                    #update the sample path and repeat
                    SP[:,:]=rand(1:nbRealization, nbSP,nbStages);
                    SP[:,1]=1;
                else
                    if (UB-LB)*1.0/max(1e-8,abs(LB)) <= eps
                        break;
                    end
                end
                #------------------------------------------
                #------------------------------------------
            end
        end
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        if PARAMS[1] == "PW_QP" || PARAMS[1] == "PW_CP"
            while true
                iter += 1;

                PartC = deepcopy(Part);
                LB, xval, thetaval, UB = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                println("iter = ", iter);
                println("========================================================================");
                println("PartsCountC=",PartsCountC);
                println("PartsCount=",PartsCount);
                println("========================================================================");
                println("LB = ", LB);
                println("UB = ", UB);
                Elapsed = time() - start - ub_calc_time;
                push!(ALL_LBS,LB)
                push!(TIMES,Elapsed)

                if Elapsed>time_limit
                    CutCount_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_",time_limit,"_CC.csv");
                    CC_df = DataFrame(ALL_cuts_counter);
                    #CSV.write(CutCount_file_name,CC_df);
                    ub_calc_stime = time();
                    nbSP_F = 10000;
                    nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
        initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                    LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                    μ = mean(UB_SP);
                    ci = 1.96*std(UB_SP)/sqrt(nbSP);
                    push!(LB_per_hr,LB)
                    push!(UB_per_hr,μ)
                    push!(CI_per_hr,ci)
                    println("time passed = ", Elapsed/(60^2))
                    println("Lower bound: ", LB);
                    println("Confidence interval: ", μ, " ± ", ci)
                    ub_calc_time = time() - ub_calc_stime
                    break
                end
                println("========================================================================");
                println("========================================================================");
                println("########################################################################");
                println(" ");

                for k=1:nbSP
                    xxval = xval[:,k,:]
                    tthetaval = thetaval[k,:]
                    for t=nbStages-1:-1:1
                        if PARAMS[1] == "PW_QP"
                            #<<<<<<<<<<PW_QP>>>>>>>>>>>>>>>
                            ###################################################################################
                            ###################################################################################

                            Part[t+1] = sort!(Part[t+1], by=length, rev=true)
                            cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                            ALL_cuts_counter[1,t]+= cut_added;
                            
                            if cut_added == 0 && length(Part[t+1]) < nbRealization
                                Part[t+1], semi_coarse_cut_counter, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time)
                                ALL_cuts_counter[2,t]+= semi_coarse_cut_counter;
                                
                            end
                        else PARAMS[1] == "PW_CP"
                            #<<<<<<<<<<PW_CP>>>>>>>>>>>>>>>
                            ###################################################################################
                            ###################################################################################
                            while true
                                semi_coarse_cut_counter = 0;
                                Part[t+1] = sort!(Part[t+1], by=length, rev=true);
                                piC = zeros(length(Part[t+1]),nbHydroP);
                                QC = zeros(length(Part[t+1]));
                                while true 
                                    cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                                    ALL_cuts_counter[1,t]+= cut_added;
                                    if cut_added == 0 
                                        break;
                                    else            
                                        range = [t];
                                        xxval, tthetaval = inner_forward_pass(MasterSubData,data,nbSP,SP,xxval,tthetaval,range,k);
                                        xval[:,k,:] = xxval;
                                        thetaval[k,:] = tthetaval;
                                    end

                                end
                                if length(Part[t+1]) < nbRealization
                                    Part[t+1], semi_coarse_cut_counter, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time)
                                    ALL_cuts_counter[2,t]+= semi_coarse_cut_counter;
                                end
                                if semi_coarse_cut_counter == 0
                                    break
                                else
                                    range = [t];
                                    xxval, tthetaval = inner_forward_pass(MasterSubData,data,nbSP,SP,xxval,tthetaval,range,k);
                                    xval[:,k,:] = xxval;
                                    thetaval[k,:] = tthetaval;
                                end

                            end
                            ###################################################################################
                            ################################################################################### 
                        end
                        PartsCountC[t+1]=length(PartC[t+1]);
                        PartsCount[t+1]=length(Part[t+1]);
                    end
                end        

                push!(LBs,LB);

                #------------------------------------------
                #------------------------------------------
                if indicator == 1
                    #update the sample path and repeat
                    SP[:,:]=rand(1:nbRealization, nbSP,nbStages);
                    SP[:,1]=1;
                else
                    if (UB-LB)*1.0/max(1e-8,abs(LB)) <= eps
                        break;
                    end
                end
                #------------------------------------------
                #------------------------------------------
            end

        end
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if PARAMS[1] == "VLVA_QP"
            while true
                iter += 1;

                PartC = deepcopy(Part);
                LB, xval, thetaval, UB = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                println("iter = ", iter);
                println("========================================================================");
                println("PartsCountC=",PartsCountC);
                println("PartsCount=",PartsCount);
                println("========================================================================");
                println("LB = ", LB);
                println("UB = ", UB);
                Elapsed = time() - start - ub_calc_time;
                push!(ALL_LBS,LB)
                push!(TIMES,Elapsed)

                if Elapsed>time_limit
                    CutCount_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_",time_limit,"_CC.csv");
                    CC_df = DataFrame(ALL_cuts_counter);
                   # CSV.write(CutCount_file_name,CC_df);
                    ub_calc_stime = time();
                    nbSP_F = 10000;
                    nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
        initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                    LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                    μ = mean(UB_SP);
                    ci = 1.96*std(UB_SP)/sqrt(nbSP);
                    push!(LB_per_hr,LB)
                    push!(UB_per_hr,μ)
                    push!(CI_per_hr,ci)
                    println("time passed = ", Elapsed/(60^2))
                    println("Lower bound: ", LB);
                    println("Confidence interval: ", μ, " ± ", ci)
                    ub_calc_time = time() - ub_calc_stime
                    break
                end
                println("========================================================================");
                println("========================================================================");
                println("########################################################################");
                println(" ");

                for k=1:nbSP
                    xxval = xval[:,k,:]
                    tthetaval = thetaval[k,:]
                    for q=length(Seasons):-1:1
                        if PARAMS[1] == "VLVA_QP"
                            #<<<<<<<<<<MLVA-QP>>>>>>>>>>>>>>>
                            ###################################################################################
                            ###################################################################################
                            for t=Seasons[q][end-1]:-1:Seasons[q][1]-1
                                Part[t+1] = sort!(Part[t+1], by=length, rev=true);
                                if q in Dry_index 
                                    #---->>>> Generate a coarse cut; if not possible; try semi-coarse
                                    cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                                    ALL_cuts_counter[1,t]+= cut_added;
                                    if cut_added == 0 && length(Part[t+1]) < nbRealization
                                        Part[t+1], semi_coarse_cut_counter, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time)
                                        ALL_cuts_counter[2,t]+= semi_coarse_cut_counter;
                                    end
                                    PartsCountC[t+1]=length(PartC[t+1]);
                                    PartsCount[t+1]=length(Part[t+1]);
                                else
                                    #---->>>> Generate a fine cut immediately
                                    cut_count = Type3_cut(MasterSubData,data,t,xval,thetaval,eps,SP,k);
                                    ALL_cuts_counter[3,t]+= cut_count;
                                    PartsCountC[t+1]=nbRealization;
                                    PartsCount[t+1]=nbRealization;
                                end
                            end
                            ###################################################################################
                            ##################################################################################                   
                        end
                    end
                    push!(LBs,LB);

                    #------------------------------------------
                    #------------------------------------------
                    if indicator == 1
                        #update the sample path and repeat
                        SP[:,:]=rand(1:nbRealization, nbSP,nbStages);
                        SP[:,1]=1;
                    else
                        if (UB-LB)*1.0/max(1e-8,abs(LB)) <= eps
                            break;
                        end
                    end
                    #------------------------------------------
                    #------------------------------------------
                end
            end
        end


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if PARAMS[1] == "PrePro" || PARAMS[1] == "Tree_Parts"
            while true
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                iter += 1;
                semi_coarse_cut_count = 0;
                PartC = deepcopy(Part);
                #LB, xval, thetaval, UB = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                println("iter = ", iter);
                println("========================================================================");
                println("PartsCountC=",PartsCountC);
                println("PartsCount=",PartsCount);
                println("========================================================================");
                println("LB = ", LB);
                println("UB = ", UB);
                Elapsed = time() - start - ub_calc_time;
                push!(ALL_LBS,LB)
                push!(TIMES,Elapsed)

                if Elapsed>time_limit
                    CutCount_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_",time_limit,"_CC.csv");
                    CC_df = DataFrame(ALL_cuts_counter);
                   # CSV.write(CutCount_file_name,CC_df);
                    ub_calc_stime = time();
                    nbSP_F = 10000;
                    nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
        initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                    LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                    μ = mean(UB_SP);
                    ci = 1.96*std(UB_SP)/sqrt(nbSP);
                    push!(LB_per_hr,LB)
                    push!(UB_per_hr,μ)
                    push!(CI_per_hr,ci)
                    println("time passed = ", Elapsed/(60^2))
                    println("Lower bound: ", LB);
                    println("Confidence interval: ", μ, " ± ", ci)
                    ub_calc_time = time() - ub_calc_stime
                    break
                end
                println("========================================================================");
                println("========================================================================");
                println("########################################################################");
                println(" ");
                #******************************************************************************************
                #******************************************************************************************
                #if the partition size is "large enough" & we are doing preprocessing revert to SDDP_QP
                if TotReal > phase3 && PARAMS[1] == "PrePro"
                    println("<<<<<<<<<<<<<< reverting back to SDDP_QP >>>>>>>>>>>");
                    nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, ignore, eps, LBs, TIMEOUT, indicator = 
                    initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                    while true

                        iter +=1
                        LB, xval, thetaval, UB = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                        println("iter = ", iter);
                        println("LB = ", LB);
                        println("UB = ", UB);
                        Elapsed = time() - start - ub_calc_time;
                        push!(ALL_LBS,LB)
                        push!(TIMES,Elapsed)

                        if Elapsed>time_limit
                            CutCount_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_",time_limit,"_CC.csv");
                            CC_df = DataFrame(ALL_cuts_counter);
                           # CSV.write(CutCount_file_name,CC_df);
                            ub_calc_stime = time();
                            nbSP_F = 10000;
                            nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
                initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                            LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                            μ = mean(UB_SP);
                            ci = 1.96*std(UB_SP)/sqrt(nbSP);
                            push!(LB_per_hr,LB)
                            push!(UB_per_hr,μ)
                            push!(CI_per_hr,ci)
                            println("time passed = ", Elapsed/(60^2))
                            println("Lower bound: ", LB);
                            println("Confidence interval: ", μ, " ± ", ci)
                            ub_calc_time = time() - ub_calc_stime
                            break
                        end
                        println("========================================================================");
                        println("========================================================================");
                        println(" ");

                        #<<<<<<<<<<SDDP_QP>>>>>>>>>>>>>>>
                        ###################################################################################
                        ###################################################################################
                        for k=1:nbSP
                            xxval = xval[:,k,:];
                            tthetaval = thetaval[k,:];
                            for t=nbStages-1:-1:1
                                cut_count = Type3_cut(MasterSubData,data,t,xval,thetaval,eps,SP,k);
                                ALL_cuts_counter[3,t]+= cut_count;
                            end
                        end
                        ###################################################################################
                        ###################################################################################
                        #------------------------------------------
                        if indicator == 1
                            #update the sample path and repeat
                            SP[:,:]=rand(1:nbRealization, nbSP,nbStages);
                            SP[:,1]=1;
                        else
                            if (UB-LB)*1.0/max(1e-8,abs(LB)) <= eps
                                break;
                            end
                        end
                        #------------------------------------------
                    end
                    break
                    #******************************************************************************************
                    #solve a MSLP on the coarse Tree & refine the partition by PW_CP once solved
                else
                    UB = 1e10
                    LB_track = [];
                    inner_iter = 0;
                    nbSP = prod(length(Part[t]) for t=1:nbStages);
                    coarse_data = [];
                    while true
                        inner_iter +=1
                        println("iter = ", iter);
                        println("LB = ", LB);
                        println("UB = ", UB);
                        push!(ALL_LBS,LB)
                        push!(TIMES,Elapsed)
                        
                        Elapsed = time() - start - ub_calc_time;
                        if Elapsed>time_limit
                            break
                        end

                        println("========================================================================");
                        println("========================================================================");
                        println(" ");


                        if nbSP > 10000 || nbSP <= 0 || indicator == 1
                            if indicator == 1 && inner_iter > 6
                                if (LB_track[inner_iter-1]-LB_track[inner_iter-4])*1.0/max(1e-8,abs(LB)) <= eps
                                    break
                                end
                            end
                            nbSP = 1;
                        else
                            if (UB-LB)*1.0/max(1e-8,abs(LB)) <= eps
                                break;
                            end
                        end
                        SP, SP_prob = RandProb(Part,nbStages,nbRealization,nbSP)
                        LB, xval, thetaval, UB = coarse_tree_FP(MasterSubData,data,nbSP,SP,xval,thetaval,LB,Part,SP_prob);
                        push!(LB_track,LB);
                        coarse_data = [];
                        #<<<<<<<<<<Coarse_Tree-SDDP_QP>>>>>>>>>>>>>>>
                        ###################################################################################
                        for k=1:nbSP
                            temp = [];
                            xxval = xval[:,k,:]
                            tthetaval = thetaval[k,:]
                            for t=nbStages-1:-1:1
                                Part[t+1] = sort!(Part[t+1], by=length, rev=true);
                                cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                                ALL_cuts_counter[1,t]+= cut_added;
                                push!(temp,(QC, piC,t));
                            end
                            push!(coarse_data,temp);
                        end
                        ###################################################################################
                    end
                    TotReal = 0;
                    for k=1:nbSP
                        xxval = xval[:,k,:]
                        tthetaval = thetaval[k,:]
                        if length(coarse_data) > 0
                            for t=nbStages-1:-1:1
                                indx = findall(x -> t in x, coarse_data[k])[1]
                                QC = coarse_data[k][indx][1]
                                piC = coarse_data[k][indx][2]
                                Part[t+1], cut_added, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time);
                                ALL_cuts_counter[2,t]+= cut_added;

                                semi_coarse_cut_count += cut_added
                                PartsCountC[t+1]=length(PartC[t+1]);
                                PartsCount[t+1]=length(Part[t+1]);
                                TotReal += PartsCount[t+1]/nbRealization
                            end
                        end
                        break
                    end
                    TotReal = TotReal/(nbStages-1);
                #******************************************************************************************
                #******************************************************************************************
                end
            end
        end        
    end    
    LBs_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_all.csv");
    LB_df = DataFrame(Any[collect(1:length(ALL_LBS)), ALL_LBS, TIMES], [:iter, :LB, :time]);
   # CSV.write(LBs_file_name,LB_df);
    
    ref_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_ref",".csv");
    ref_df = DataFrame(Any[collect(1:length(Refinement_time)), Refinement_time], [:iter, :time]);
   # CSV.write(ref_file_name,ref_df);
    
    println("**********************************************************************************");
    println("**********************************************************************************");
    per_hr_file_name = string("./24T_50R_48hrs/",PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_per_hr.csv");
    HOURS_n = [1, 3, 6, 12, 24, 48];
    per_hr_df = DataFrame(Any[HOURS_n, LB_per_hr, UB_per_hr, CI_per_hr], [:hr, :LB_per_hr, :UB_per_hr, :CI_per_hr]);
    CSV.write(per_hr_file_name,per_hr_df);
end

@time ff()






#########################################################################################################################
#########################################################################################################################
