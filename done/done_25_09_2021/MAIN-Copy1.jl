
PARAMS = ARGS;
TIMEOUT = 60*60*6;
time_limit = 60*60*6;
srand(2019);


wid = addprocs(1)[1];
############################################################
include("packages.jl");
include("problem_data.jl");
include("model.jl");
include("auxiliary_functions-Copy1.jl");
############################################################


function ff()
    start=time();
    ALL_LBS = [];
    TIMES = [];
    Refinement_time = [];
    LB_per_hr = [];
    UB_per_hr = [];
    CI_per_hr = [];
    
    #nbCuts_perStage_perType_perHour = [];
    nbSP_F = 1;
    nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
    initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if PARAMS[1] == "SDDP_QP" || PARAMS[1] == "SDDP_HP" || PARAMS[1] == "SDDP_CP" || PARAMS[1] == "SDDP_HP_eps"
        
        if PARAMS[1] == "SDDP_HP"
            Seasons, Wet_index, Dry_index, Wet, Dry = seasons_class(MasterSubData,data,PARAMS);
        end
        while true
            iter += 1;
            LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);

            println("iter = ", iter);
            println("LB = ", LB);
            println("UB = ", UB);
            Elapsed = time() - start;
            push!(ALL_LBS,LB)
            push!(TIMES,Elapsed)
            
            
            if Elapsed>time_limit
                LBs_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],".csv");
                LB_df = DataFrame(Any[collect(1:iter), ALL_LBS, TIMES], [:iter, :LB, :time]);
                CSV.write(LBs_file_name,LB_df);
                println("**********************************************************************************");
                println("**********************************************************************************");
                println("**********************************************************************************");
                println("**********************************************************************************");
                nbSP_F = 10000;
                nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
    initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                μ = mean(UB_SP);
                ci = 1.96*std(UB_SP)/sqrt(nbSP);
                println("Lower bound: ", LB);
                println("Confidence interval: ", μ, " ± ", ci)
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
                    end
                end
                ###################################################################################
                ###################################################################################
            elseif PARAMS[1] == "SDDP_HP"
                #<<<<<<<<<<SDDP_HP>>>>>>>>>>>>>>>
                ###################################################################################
                ###################################################################################
                for k=1:nbSP
                    xxval = xval[:,k,:]
                    tthetaval = thetaval[k,:]
                    for q=length(Seasons):-1:1
                        while true
                            cut_count = 0;
                            for t=Seasons[q][end-1]:-1:Seasons[q][1]-1
                                cut_count += Type3_cut(MasterSubData,data,t,xval,thetaval,eps,SP,k);
                            end
                            if cut_count == 0 
                                break;
                            else            
                                range = [];
                                push!(range,Seasons[q][1]-1)
                                range = vcat(range,Seasons[q])
                                xxval, tthetaval = inner_forward_pass(MasterSubData,data,nbSP,SP,xxval,tthetaval,range,k);
                                xval[:,k,:] = xxval;
                                thetaval[k,:] = tthetaval;
                            end
                        end
                    end
                end
                ###################################################################################
                ###################################################################################  

            elseif PARAMS[1] == "SDDP_CP"
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
                ###################################################################################
                ###################################################################################  
           elseif PARAMS[1] == "SDDP_HP_eps"
                #<<<<<<<<<<SDDP_HP_eps>>>>>>>>>>>>>>>
                ###################################################################################
                ###################################################################################
                for k=1:nbSP
                    xxval = xval[:,k,:]
                    tthetaval = thetaval[k,:]
                    for t=nbStages-1:-1:1
                        while true
			    ϵ=1e-1;
			    Reps = (1/iter)*(ϵ-((ϵ-eps)/(nbStages-2))*(t-2));
                            cut_count = 0;
                            cut_count += Type3_cut(MasterSubData,data,t,xval,thetaval,Reps,SP,k);

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
                ###################################################################################
                ################################################################################### 

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
    if PARAMS[1] == "PW_QP" || PARAMS[1] == "PW_HP" || PARAMS[1] == "PW_CP"
        PartsCount = ones(nbStages); #array to keep track the size of partition at every time period 
        PartsCountC = ones(nbStages);
        Part = initial_Partition(nbRealization,nbStages); 

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
            Elapsed = time() - start;
            push!(ALL_LBS,LB)
            push!(TIMES,Elapsed)
            
            if Elapsed>time_limit
                LBs_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],".csv");
                LB_df = DataFrame(Any[collect(1:iter), ALL_LBS, TIMES], [:iter, :LB, :time])
                CSV.write(LBs_file_name,LB_df); 
                    
                ref_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_ref",".csv");
                ref_df = DataFrame(Any[collect(1:length(Refinement_time)), Refinement_time], [:iter, :time]);
                CSV.write(ref_file_name,ref_df);
                println("**********************************************************************************");
                println("**********************************************************************************");
                println("**********************************************************************************");
                println("**********************************************************************************");
                nbSP_F = 10000;
                nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
    initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                μ = mean(UB_SP);
                ci = 1.96*std(UB_SP)/sqrt(nbSP);
                println("Lower bound: ", LB);
                println("Confidence interval: ", μ, " ± ", ci)
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
                        if cut_added == 0 && length(Part[t+1]) < nbRealization
                            Part[t+1], semi_coarse_cut_counter, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time)
                        end
                        ###################################################################################
                        ###################################################################################
                    elseif PARAMS[1] == "PW_HP"
                        #<<<<<<<<<<PW_HP>>>>>>>>>>>>>>>
                        ###################################################################################
                        ###################################################################################
                        Part[t+1] = sort!(Part[t+1], by=length, rev=true);
                        piC = zeros(length(Part[t+1]),nbHydroP);
                        QC = zeros(length(Part[t+1]));
                        while true 
                            cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
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
                        end
                        ###################################################################################
                        ###################################################################################
                    elseif PARAMS[1] == "PW_CP"
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if PARAMS[1] == "VLVA_QP" ||
       PARAMS[1] == "MLMA_HP" || PARAMS[1] == "VLVA_HP"||
       PARAMS[1] == "MLMA_CP" || PARAMS[1] == "VLVA_CP"
        Seasons, Wet_index, Dry_index, Wet, Dry = seasons_class(MasterSubData,data,PARAMS);
        PartsCount = ones(nbStages); #array to keep track the size of partition at every time period 
        PartsCountC = ones(nbStages);
        Part = initial_Partition(nbRealization,nbStages); 

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
            Elapsed = time() - start;
            push!(ALL_LBS,LB)
            push!(TIMES,Elapsed)
            
            if Elapsed>time_limit
                LBs_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],".csv");
                LB_df = DataFrame(Any[collect(1:iter), ALL_LBS, TIMES], [:iter, :LB, :time])
                CSV.write(LBs_file_name,LB_df); 
                    
                ref_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_ref",".csv");
                ref_df = DataFrame(Any[collect(1:length(Refinement_time)), Refinement_time], [:iter, :time]);
                CSV.write(ref_file_name,ref_df);
                    
                println("**********************************************************************************");
                println("**********************************************************************************");
                println("**********************************************************************************");
                println("**********************************************************************************");
                nbSP_F = 10000;
                nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
    initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                μ = mean(UB_SP);
                ci = 1.96*std(UB_SP)/sqrt(nbSP);
                println("Lower bound: ", LB);
                println("Confidence interval: ", μ, " ± ", ci)
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
                                if cut_added == 0 && length(Part[t+1]) < nbRealization
                                    Part[t+1], semi_coarse_cut_counter, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time)
                                end
                                PartsCountC[t+1]=length(PartC[t+1]);
                                PartsCount[t+1]=length(Part[t+1]);
                            else
                                #---->>>> Generate a fine cut immediately
                                cut_count = Type3_cut(MasterSubData,data,t,xval,thetaval,eps,SP,k);
                                PartsCountC[t+1]=nbRealization;
                                PartsCount[t+1]=nbRealization;
                            end
                        end
                        ###################################################################################
                        ###################################################################################
                    elseif PARAMS[1] == "MLMA_HP"
                        #<<<<<<<<<<MLMA-HP>>>>>>>>>>>>>>>
                        ###################################################################################
                        ###################################################################################
                          while true
                            #---->>>> Regardless of the season, Generate a coarse cut
                            cut_count = 0;
                            coarse_data = [];
                            for t=Seasons[q][end-1]:-1:Seasons[q][1]-1
                                Part[t+1] = sort!(Part[t+1], by=length, rev=true);
                                cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                                cut_count += cut_added;
                                push!(coarse_data,(QC, piC,t));
                            end
                            if cut_count == 0;
                                #depending on the season generate semi-coarse/fine cut
                                for t=Seasons[q][end-1]:-1:Seasons[q][1]-1
                                    if length(Part[t+1]) < nbRealization
                                        indx = findall(x -> t in x, coarse_data)[1]
                                        QC = coarse_data[indx][1]
                                        piC = coarse_data[indx][2]
                                    
                                        if q in Dry_index
                                            #---->>>> Generate a semi-coarse cut
                                            Part[t+1], semi_coarse_cut_counter, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time)
                                        else
                                            #---->>>> Generate a fine cut immediately & refine the partition
                                            Part[t+1], fine_cut_counter = Type4_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,Refinement_time);
                                        end
                                        PartsCountC[t+1]=length(PartC[t+1]);
                                        PartsCount[t+1]=length(Part[t+1]);
                                    end
                                end
                                break;
                            else            
                                range = [];
                                push!(range,Seasons[q][1]-1)
                                range = vcat(range,Seasons[q])
                                xxval, tthetaval = inner_forward_pass(MasterSubData,data,nbSP,SP,xxval,tthetaval,range,k);
                                xval[:,k,:] = xxval;
                                thetaval[k,:] = tthetaval;
                            end
                        end
                    ###################################################################################
                    ###################################################################################
                    elseif PARAMS[1] == "VLVA_HP"
                        #<<<<<<<<<<VLVA-HP>>>>>>>>>>>>>>>
                        ###################################################################################
                        ###################################################################################
                        while true
                            
                            #---->>>> Regardless of the season, Generate a coarse cut
                            cut_count = 0;
                            coarse_data = [];
                            for t=Seasons[q][end-1]:-1:Seasons[q][1]-1
                                Part[t+1] = sort!(Part[t+1], by=length, rev=true);
                                cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                                if q in Dry_index && cut_added == 0 && length(Part[t+1]) < nbRealization
                                    #try semi-coarse
                                    Part[t+1], semi_coarse_cut_counter, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time)
                                    PartsCountC[t+1]=length(PartC[t+1]);
                                    PartsCount[t+1]=length(Part[t+1]);
                                end
                                cut_count += cut_added;
                                push!(coarse_data,(QC, piC,t));

                            end
                            if q in Dry_index 
                                break;
                            elseif q in Wet_index && cut_count == 0;
                                #generate all semi_coarse
                                semi_coarse_cut_counter = 0;
                                for t=Seasons[q][end-1]:-1:Seasons[q][1]-1
                                    if length(Part[t+1]) < nbRealization
                                        indx = findall(x -> t in x, coarse_data)[1]
                                        QC = coarse_data[indx][1]
                                        piC = coarse_data[indx][2]
                                        Part[t+1], cut_added, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time)
                                        semi_coarse_cut_counter += cut_added
                                    end
                                    PartsCountC[t+1]=length(PartC[t+1]);
                                    PartsCount[t+1]=length(Part[t+1]);
                                end
                                if semi_coarse_cut_counter == 0
                                    break;
                                end
                            end
                            range = [];
                            push!(range,Seasons[q][1]-1)
                            range = vcat(range,Seasons[q])
                            xxval, tthetaval = inner_forward_pass(MasterSubData,data,nbSP,SP,xxval,tthetaval,range,k);
                            xval[:,k,:] = xxval;
                            thetaval[k,:] = tthetaval;

                        end
                    ###################################################################################
                    ###################################################################################
                    elseif PARAMS[1] == "MLMA_CP"
                         #<<<<<<<<<<MLMA-CP>>>>>>>>>>>>>>>
                        ###################################################################################
                        ###################################################################################
                        for t=Seasons[q][end-1]:-1:Seasons[q][1]-1
                            while true
                                Part[t+1] = sort!(Part[t+1], by=length, rev=true);
                                cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                                #---->>>> #try semi-coarse
                                if q in Dry_index && cut_added == 0 && length(Part[t+1]) < nbRealization
                                    Part[t+1], cut_added, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time);
                                    break
                                else
                                    cut_count = Type3_cut(MasterSubData,data,t,xval,thetaval,eps,SP,k);
                                    if cut_count == 0
                                        break
                                    end
                                end
                                range = [t];
                                xxval, tthetaval = inner_forward_pass(MasterSubData,data,nbSP,SP,xxval,tthetaval,range,k);
                                xval[:,k,:] = xxval;
                                thetaval[k,:] = tthetaval;
                            end
                            PartsCountC[t+1]=length(PartC[t+1]);
                            PartsCount[t+1]=length(Part[t+1]);
                        end
                        ###################################################################################
                        ###################################################################################  
                    elseif PARAMS[1] == "VLVA_CP"
                    
                        #<<<<<<<<<<VLVA-CP>>>>>>>>>>>>>>>
                        ###################################################################################
                        ###################################################################################
                        for t=Seasons[q][end-1]:-1:Seasons[q][1]-1
                            while true
                                if q in Dry_index
                                    Part[t+1] = sort!(Part[t+1], by=length, rev=true);
                                    cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                                    #---->>>> #try semi-coarse
                                    if cut_added == 0 
                                        if length(Part[t+1]) < nbRealization
                                            Part[t+1], cut_added, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time);
                                            PartsCountC[t+1]=length(PartC[t+1]);
                                            PartsCount[t+1]=length(Part[t+1]);
                                        end
                                        break
                                    end
                                else
                                    #---->>>> Generate a fine cut immediately
                                    cut_count = Type3_cut(MasterSubData,data,t,xval,thetaval,eps,SP,k);
                                    if cut_count == 0
                                        PartsCountC[t+1]=nbRealization;
                                        PartsCount[t+1]=nbRealization
                                        break
                                    end
                                end
                                range = [t];
                                xxval, tthetaval = inner_forward_pass(MasterSubData,data,nbSP,SP,xxval,tthetaval,range,k);
                                xval[:,k,:] = xxval;
                                thetaval[k,:] = tthetaval;
                            end
                        end
                        ###################################################################################
                        ###################################################################################                     
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if PARAMS[1] == "PrePro" || PARAMS[1] == "Tree_Parts"
        phase3 = 0.5
        PartsCount = ones(nbStages); #array to keep track the size of partition at every time period 
        PartsCountC = ones(nbStages);
        Part = initial_Partition(nbRealization,nbStages); 
        TotReal = sum(PartsCount[t]/nbRealization for t=2:nbStages)/(nbStages-1);
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
            Elapsed = time() - start;
            push!(ALL_LBS,LB)
            push!(TIMES,Elapsed)
            
            if Elapsed>time_limit
                LBs_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],".csv");
                LB_df = DataFrame(Any[collect(1:length(ALL_LBS)), ALL_LBS, TIMES], [:iter, :LB, :time])
                CSV.write(LBs_file_name,LB_df); 
                    
                ref_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_ref",".csv");
                ref_df = DataFrame(Any[collect(1:length(Refinement_time)), Refinement_time], [:iter, :time]);
                CSV.write(ref_file_name,ref_df);
                    
                println("**********************************************************************************");
                println("**********************************************************************************");
                println("**********************************************************************************");
                println("**********************************************************************************");
                nbSP_F = 10000;
                nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
    initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                μ = mean(UB_SP);
                ci = 1.96*std(UB_SP)/sqrt(nbSP);
                println("Lower bound: ", LB);
                println("Confidence interval: ", μ, " ± ", ci)
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
                initialize(nbRealization,nbStages,nbHydroP);
                while true
                
                    iter +=1
                    LB, xval, thetaval, UB = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                    println("iter = ", iter);
                    println("LB = ", LB);
                    println("UB = ", UB);
                    Elapsed = time() - start;
                    push!(ALL_LBS,LB)
                    push!(TIMES,Elapsed)

                    if Elapsed>time_limit
                        LBs_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],".csv");
                        LB_df = DataFrame(Any[collect(1:length(ALL_LBS)), ALL_LBS, TIMES], [:iter, :LB, :time])
                        CSV.write(LBs_file_name,LB_df); 
                            
                        ref_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_ref",".csv");
                        ref_df = DataFrame(Any[collect(1:length(Refinement_time)), Refinement_time], [:iter, :time]);
                        CSV.write(ref_file_name,ref_df);
                            
                        println("**********************************************************************************");
                        println("**********************************************************************************");
                        println("**********************************************************************************");
                        println("**********************************************************************************");
                        nbSP_F = 10000;
                        nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
            initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                        LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                        μ = mean(UB_SP);
                        ci = 1.96*std(UB_SP)/sqrt(nbSP);
                        println("Lower bound: ", LB);
                        println("Confidence interval: ", μ, " ± ", ci)
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
                    Elapsed = time() - start;
                    push!(ALL_LBS,LB)
                    push!(TIMES,Elapsed)

                    if Elapsed>time_limit
                        LBs_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],".csv");
                        LB_df = DataFrame(Any[collect(1:length(ALL_LBS)), ALL_LBS, TIMES], [:iter, :LB, :time])
                        CSV.write(LBs_file_name,LB_df); 
                            
                        ref_file_name = string(PARAMS[1],"_",PARAMS[2],"_",PARAMS[3],"_ref",".csv");
                        ref_df = DataFrame(Any[collect(1:length(Refinement_time)), Refinement_time], [:iter, :time]);
                        CSV.write(ref_file_name,ref_df);
                            
                        println("**********************************************************************************");
                        println("**********************************************************************************");
                        println("**********************************************************************************");
                        println("**********************************************************************************");
                        nbSP_F = 10000;
                        nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
            initialize(nbRealization,nbStages,nbHydroP,nbSP_F);
                        LB, xval, thetaval, UB, UB_SP = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                        μ = mean(UB_SP);
                        ci = 1.96*std(UB_SP)/sqrt(nbSP);
                        println("Lower bound: ", LB);
                        println("Confidence interval: ", μ, " ± ", ci)
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
                    for t=nbStages-1:-1:1
                        indx = findall(x -> t in x, coarse_data[k])[1]
                        QC = coarse_data[k][indx][1]
                        piC = coarse_data[k][indx][2]
                        Part[t+1], cut_added, Refinement_time = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC,Refinement_time);
                        semi_coarse_cut_count += cut_added
                        PartsCountC[t+1]=length(PartC[t+1]);
                        PartsCount[t+1]=length(Part[t+1]);
                        TotReal += PartsCount[t+1]/nbRealization
                    end
                    break
                end
                TotReal = TotReal/(nbStages-1);
            #******************************************************************************************
            #******************************************************************************************
            end

            
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end
       
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   
end

ff()

#########################################################################################################################
#########################################################################################################################
