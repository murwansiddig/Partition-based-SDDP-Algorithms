
PARAMS = ARGS;
TIMEOUT = 60*60*6;
srand(2019);


wid = addprocs(1)[1];
############################################################
include("packages.jl");
include("problem_data.jl");
include("model.jl");
include("auxiliary_functions.jl");
############################################################


@everywhere function
ff(Refine, PARAMS, MasterSubData, data, MasterSub, xvar, yvar,var, gvar, pvar,
   thetavar, FBconstr, MVPobj, scen, nbStages, nbRealization,nbHydroP, nbThermoP, cp, demand,
   qbar, vlow, vup, x0, rh, fbar, cf, c0, HReserv,HNoReserv, Graph, Type1_cut, Type2_cut, Type3_cut,Type4_cut,
   forward_pass, inner_forward_pass,seasons_class,NewScen,AllSP,RandProb,initialize,initial_Partition)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, iter, eps, LBs, TIMEOUT, indicator = 
    initialize(nbRealization,nbStages,nbHydroP);
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if PARAMS[1] == "SDDP-QP" || PARAMS[1] == "SDDP-CP"
        
        while true
            iter += 1;
            LB, xval, thetaval, UB = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);

            println("iter = ", iter);
            println("LB = ", LB);
            println("UB = ", UB);
            println("========================================================================");
            println("========================================================================");
            println(" ");

            if PARAMS[1] == "SDDP-QP"
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

            elseif PARAMS[1] == "SDDP-CP"
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
    if PARAMS[1] == "APQP-SDDP" || PARAMS[1] == ""APCP-SDDP""
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
            println("========================================================================");
            println("========================================================================");
            println("########################################################################");
            println(" ");

            for k=1:nbSP
                xxval = xval[:,k,:]
                tthetaval = thetaval[k,:]
                for t=nbStages-1:-1:1
                    if PARAMS[1] == "APQP-SDDP"
                        #<<<<<<<<<<PW_QP>>>>>>>>>>>>>>>
                        ###################################################################################
                        ###################################################################################

                        Part[t+1] = sort!(Part[t+1], by=length, rev=true)
                        cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                        if cut_added == 0 && length(Part[t+1]) < nbRealization
                            Part[t+1], semi_coarse_cut_counter = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC)
                        end
                        ###################################################################################
                        ###################################################################################
                    elseif PARAMS[1] == ""APCP-SDDP""
                        #<<<<<<<<<<"APCP-SDDP">>>>>>>>>>>>>>>
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
                                Part[t+1], semi_coarse_cut_counter = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC)
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
    if PARAMS[1] == "SPAP-SDDP"
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
            println("========================================================================");
            println("========================================================================");
            println("########################################################################");
            println(" ");

            for k=1:nbSP
                xxval = xval[:,k,:]
                tthetaval = thetaval[k,:]
                for q=length(Seasons):-1:1
                ###################################################################################
                ###################################################################################
                for t=Seasons[q][end-1]:-1:Seasons[q][1]-1
                    Part[t+1] = sort!(Part[t+1], by=length, rev=true);
                    if q in Dry_index 
                        #---->>>> Generate a coarse cut; if not possible; try semi-coarse
                        cut_added, QC, piC = Type1_cut(MasterSubData,data,Part,SPC,t,xxval,tthetaval,eps,SP,k);
                        if cut_added == 0 && length(Part[t+1]) < nbRealization
                            Part[t+1], semi_coarse_cut_counter = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC)
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
                end
            end

        end
    end
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if PARAMS[1] == "APEP-SDDP" || PARAMS[1] == "ITER-SDDP"
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
            println("========================================================================");
            println("========================================================================");
            println("########################################################################");
            println(" ");
            #******************************************************************************************
            #******************************************************************************************
            #if the partition size is "large enough" & we are doing preprocessing revert to SDDP_QP
            if TotReal > phase3 && PARAMS[1] == "APEP-SDDP"
                println("<<<<<<<<<<<<<< reverting back to SDDP_QP >>>>>>>>>>>");
                nbSP, SP, objval, xval, thetaval, UB_SP, optimalx_1, ETGF, LB, Local_UB, UB, ignore, eps, LBs, TIMEOUT, indicator = 
                initialize(nbRealization,nbStages,nbHydroP);
                while true
                
                    iter +=1
                    LB, xval, thetaval, UB = forward_pass(MasterSubData,data,nbSP,SP,xval,thetaval,LB);
                    println("iter = ", iter);
                    println("LB = ", LB);
                    println("UB = ", UB);
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
                #solve a MSLP on the coarse Tree & refine the partition by "APCP-SDDP" once solved
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
                        Part[t+1], cut_added = Type2_cut(MasterSubData,Part,SPC,SPSC,data,t,xxval,tthetaval,Refine,eps,SP,k,QC,piC);
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

run_with_timeout(TIMEOUT,() ->(try;
        ff(Refine, PARAMS, MasterSubData, data, MasterSub, xvar, yvar,var, gvar, pvar,
        thetavar, FBconstr, MVPobj, scen, nbStages, nbRealization,nbHydroP, nbThermoP, cp, demand,
        qbar, vlow, vup, x0, rh, fbar, cf, c0, HReserv,HNoReserv, Graph, Type1_cut, Type2_cut, Type3_cut,Type4_cut,
        forward_pass, inner_forward_pass,seasons_class,NewScen,AllSP,RandProb,initialize,initial_Partition);
        catch ee;dump(ee);end),wid);

#########################################################################################################################
#########################################################################################################################
