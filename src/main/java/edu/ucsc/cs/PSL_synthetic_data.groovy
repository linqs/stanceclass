package edu.ucsc.cs


import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.random.GroundSliceRandOM;
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxmargin.PositiveMinNormProgram;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.VotedPerceptron;
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM
import edu.umd.cs.psl.application.learning.weight.em.DualEM
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;
import edu.umd.cs.psl.evaluation.statistics.RankingScore;
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter




class PSL_synthetic_data{

	public static void main(String[] args)
	{
		for(int i = 0; i <1; ++i)
		{
			PSL_synthetic_data a = new PSL_synthetic_data()
			a.pslmodel(i);
		}
	}

	void pslmodel(int cvSet)
	{
		/*
		 * Config bundle changed to accept String as UniqueID
		 */
		ConfigManager cm = ConfigManager.getManager()
		ConfigBundle config = cm.getBundle("stance-classification")
		String writefolder = System.getProperty("user.home") + "/Documents/Shachi/CMPS209C/reviews/Results/Synthetic_data/"
		File file3 = new File(writefolder+"results.csv");


		/* Uses H2 as a DataStore and stores it in a temp. directory by default */
		def defaultPath = System.getProperty("java.io.tmpdir")
		String dbpath = config.getString("dbpath", defaultPath + File.separator + "stance-classification")
		DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

		/*
		 * Initialize PSL model
		 */
		PSLModel m = new PSLModel(this, data)


		/*
		 * Predicates
		 */
		m.add predicate: "prev" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "contrast" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "noncontrast", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "priorpos", types: [ArgumentType.UniqueID]
		m.add predicate: "priorneg", types: [ArgumentType.UniqueID]
		m.add predicate: "subjectivitypos", types: [ArgumentType.UniqueID]
		m.add predicate: "subjectivityneg", types: [ArgumentType.UniqueID]
		m.add predicate: "possentiment", types: [ArgumentType.UniqueID]
		m.add predicate: "negsentiment", types: [ArgumentType.UniqueID]
		m.add predicate: "all", types: [ArgumentType.UniqueID]
		m.add predicate: "nrclexiconpos", types: [ArgumentType.UniqueID]
		m.add predicate: "nrclexiconneg", types: [ArgumentType.UniqueID]
		m.add predicate: "unigrampos", types: [ArgumentType.UniqueID]
		m.add predicate: "unigramneg", types: [ArgumentType.UniqueID]
		m.add predicate: "tglpos", types: [ArgumentType.UniqueID]
		m.add predicate: "tglneg", types: [ArgumentType.UniqueID]


		/*
		 * Adding rules
		 */

		/*
		 * Rules for attribute features alone - sentiment lexicons as source
		 */
		m.add rule : (possentiment(A) ) >> negsentiment(A), weight :5, squared : true
		//m.add rule : (negsentiment(A) ) >> ~possentiment(A), weight :5
		m.add rule : (negsentiment(A)) >> possentiment(A), weight :5, squared : true

		m.add rule : (priorneg(A) ) >> possentiment(A), weight :5, squared : true
		m.add rule : (priorpos(A) ) >> negsentiment(A), weight :5, squared : true

//		m.add rule : possentiment(A) >> (priorpos(A) ) , weight :5, squared : false
//		m.add rule : negsentiment(A) >> (priorneg(A) ) , weight :5, squared : false
		
//		m.add rule : possentiment(A) >> (priorpos(A) ) , weight :5
//		m.add rule : negsentiment(A) >> (priorneg(A) ) , weight :5

//		m.add rule : (all(A) & ~(priorpos(A) )) >> ~possentiment(A), weight :5, squared : false
//		m.add rule : (all(A) & ~(priorneg(A) )) >> ~negsentiment(A), weight :5, squared : false


//		m.add rule : subjectivitypos(A) >> possentiment(A), weight : 5, squared : false
//		m.add rule : subjectivityneg(A) >> negsentiment(A), weight : 5, squared : false
//
//		//m.add rule : possentiment(A) >> (subjectivitypos(A) ) , weight :5
//		//m.add rule : negsentiment(A) >> (subjectivityneg(A) ) , weight :5
//
//		//m.add rule : (~subjectivitypos(A)) >> ~possentiment(A), weight : 5
//		//m.add rule : ( ~subjectivityneg(A)) >> ~negsentiment(A), weight : 5
//
//		m.add rule : nrclexiconpos(A) >> possentiment(A), weight : 5, squared : false
//		m.add rule : nrclexiconneg(A) >> negsentiment(A), weight : 5, squared : false
////
//		m.add rule : possentiment(A) >> (nrclexiconpos(A) ) , weight :5, squared :false
//		m.add rule : negsentiment(A) >> (nrclexiconneg(A) ) , weight :5, squared :false
//
//
////		m.add rule : (all(A) & ~nrclexiconpos(A)) >> ~possentiment(A), weight : 5, squared : false
////		m.add rule : (all(A) & ~nrclexiconneg(A)) >> ~negsentiment(A), weight : 5, squared : false
//
//
//		m.add rule : unigrampos(A) >> possentiment(A), weight : 5, squared : false
//		m.add rule : unigramneg(A) >> negsentiment(A), weight : 5, squared : false
//
//		//m.add rule : possentiment(A) >> (unigrampos(A) ) , weight :5
//		//m.add rule : negsentiment(A) >> (unigramneg(A) ) , weight :5
//
//
//		//m.add rule : ( ~unigrampos(A)) >> ~possentiment(A), weight : 5
//		//m.add rule : ( ~unigramneg(A)) >> ~negsentiment(A), weight : 5
//
//
		m.add rule : tglpos(A) >> possentiment(A), weight : 5, squared : true
		m.add rule : tglneg(A) >> negsentiment(A), weight : 5, squared : true
//
//		//m.add rule : possentiment(A) >> (tglpos(A) ) , weight :5
//		//m.add rule : negsentiment(A) >> (tglneg(A) ) , weight :5
//
//
//		//m.add rule : (~tglpos(A)) >> ~possentiment(A), weight : 5
//		//m.add rule : (~tglneg(A)) >> ~negsentiment(A), weight : 5
//

		/*  
		 * Rules for Neighborhood relation
		 * 
		 * 
		 */
		m.add rule : (prev(A,B) & possentiment(B)) >> possentiment(A), weight :5, squared : true
		m.add rule : (prev(A,B) & negsentiment(B)) >> negsentiment(A), weight :5, squared : true

		/*
		 * Rules for contrast and non-contrast relation
		 */
//		m.add rule : (contrast(A,B) & possentiment(B) ) >> negsentiment(A)  , weight :10, squared : false
//		m.add rule : (contrast(A,B) & negsentiment(B) ) >> possentiment(A)  , weight :10, squared : false
//
//		m.add rule : (noncontrast(A,B) & possentiment(B) ) >> possentiment(A)  , weight :10, squared : false
//		m.add rule : (noncontrast(A,B) & negsentiment(B) ) >> negsentiment(A)  , weight :10, squared : false


		int folds = 10
		/*
		 * The results are shown for all threshold levels.
		 */

		/*
		 * There is some issue with the cross validation looping code, so currently have to set each cvSet manually and run for each fold.
		 */

		//for(cvSet =0 ;cvSet<10;++cvSet)
		//{
		List<Partition> trainPartition = new ArrayList<Partition>(folds)
		List<Partition> trueDataPartition = new ArrayList<Partition>(folds)
		List<Partition> testDataPartition = new ArrayList<Partition>(folds)
		List<Partition> trueTestDataPartition = new ArrayList<Partition>(folds)


		/*
		 * Initialize partitions for all cross validation sets
		 */


		for(int initset =0 ;initset<10;++initset)
		{
			trainPartition.add(initset, new Partition(initset))
			trueDataPartition.add(initset, new Partition(initset + folds))
			testDataPartition.add(initset, new Partition(initset + 2*folds))
			trueTestDataPartition.add(initset, new Partition(initset + 3*folds))
		}



		/*
		 * Set the cross validation fold set
		 */
		//cvSet = 9
		/*
		 * Set the folder to write into
		 */
		Integer folder = (cvSet+10)%10;
		if (folder ==0) folder = 10
		String filename1 = writefolder+"fold"+folder+"/possentiment.csv"
		String filename2 = writefolder+"fold"+folder+"/negsentiment.csv"
		File file1 = new File(filename1);
		File file2 = new File(filename2);
		File file4 = new File(writefolder+"auc.csv");
		File file5 = new File(writefolder+"model.csv");
		/*
		 * Train data partition, each partition has 9 folders, one kept aside for testing... 
		 * 
		 * loading the predicates from the data files into the trainPartition
		 */
		String filename
		Integer trainSet
//		for (trainSet = 1 ; trainSet<=9;++trainSet)
//		{
//			Integer dirToUse = 0;
//			dirToUse = (cvSet+trainSet)%10
//			if(dirToUse==0) dirToUse = 10;

			filename = 'data'+java.io.File.separator+'sentiment/synthetic_data/train/'
//			InserterUtils.loadDelimitedDataTruth(data.getInserter(nrclexiconpos, trainPartition.get(cvSet)),
//					filename+"NRC_negation_pos_changed.csv","\t");
//			InserterUtils.loadDelimitedDataTruth(data.getInserter(nrclexiconneg, trainPartition.get(cvSet)),
//					filename+"NRC_negation_neg_changed.csv","\t");
//			InserterUtils.loadDelimitedDataTruth(data.getInserter(unigrampos, trainPartition.get(cvSet)),
//					filename+"unigram_pos_negation_changed.csv","\t");
//			InserterUtils.loadDelimitedDataTruth(data.getInserter(unigramneg, trainPartition.get(cvSet)),
//					filename+"unigram_neg_negation_changed.csv","\t");
//
			InserterUtils.loadDelimitedDataTruth(data.getInserter(tglpos, trainPartition.get(cvSet)),
					filename+"pos_tgl.csv","\t");
			InserterUtils.loadDelimitedDataTruth(data.getInserter(tglneg, trainPartition.get(cvSet)),
					filename+"neg_tgl.csv","\t");
//
			InserterUtils.loadDelimitedData(data.getInserter(prev, trainPartition.get(cvSet)), filename+"all_prev.csv");
//
//
			InserterUtils.loadDelimitedDataTruth(data.getInserter(priorpos, trainPartition.get(cvSet)),
					filename+"pos_swn.csv","\t");
			InserterUtils.loadDelimitedDataTruth(data.getInserter(priorneg, trainPartition.get(cvSet)),
					filename+"neg_swn.csv","\t");
			InserterUtils.loadDelimitedData(data.getInserter(all, trainPartition.get(cvSet)), filename+"allID.csv");
//
//			InserterUtils.loadDelimitedData(data.getInserter(contrast, trainPartition.get(cvSet)),
//					filename+"contrast_ids.csv");
//			InserterUtils.loadDelimitedData(data.getInserter(noncontrast, trainPartition.get(cvSet)),
//					filename+"noncontrast_ids.csv");

			/*
			 * Load in the ground truth positive and negative segments
			 */
			InserterUtils.loadDelimitedData(data.getInserter(negsentiment, trueDataPartition.get(cvSet)), 
				filename+"trueneg.csv");
			InserterUtils.loadDelimitedData(data.getInserter(possentiment, trueDataPartition.get(cvSet)), 
				filename+"truepos.csv");
//		}


		/*
		 * For test data partition - it needs only one fold in each partition.... Start with 10,1,2,3.... so on. 
		 */
		Integer testSet = 0;
		testSet = (cvSet+10)%10
		if(testSet==0) testSet = 10;
		filename = 'data'+java.io.File.separator+'sentiment/synthetic_data/test/';

		InserterUtils.loadDelimitedData(data.getInserter(prev, testDataPartition.get(cvSet)), filename+"all_prev.csv");

//
//		InserterUtils.loadDelimitedDataTruth(data.getInserter(unigrampos,
//				testDataPartition.get(cvSet)), filename+"unigram_pos_negation_changed.csv");
//
//		InserterUtils.loadDelimitedDataTruth(data.getInserter(unigramneg,
//				testDataPartition.get(cvSet)), filename+"unigram_neg_negation_changed.csv");
//
//		InserterUtils.loadDelimitedDataTruth(data.getInserter(nrclexiconpos, testDataPartition.get(cvSet)),
//				filename+"NRC_negation_pos_changed.csv","\t");
//		InserterUtils.loadDelimitedDataTruth(data.getInserter(nrclexiconneg, testDataPartition.get(cvSet)),
//				filename+"NRC_negation_neg_changed.csv","\t");
		InserterUtils.loadDelimitedDataTruth(data.getInserter(priorpos, testDataPartition.get(cvSet)),
				filename+"pos_swn.csv","\t");

		InserterUtils.loadDelimitedDataTruth(data.getInserter(priorneg, testDataPartition.get(cvSet)),
				filename+"neg_swn.csv","\t");

		InserterUtils.loadDelimitedDataTruth(data.getInserter(tglpos, testDataPartition.get(cvSet)),
				filename+"pos_tgl.csv","\t");
		InserterUtils.loadDelimitedDataTruth(data.getInserter(tglneg, testDataPartition.get(cvSet)),
				filename+"neg_tgl.csv","\t");

		InserterUtils.loadDelimitedData(data.getInserter(all, testDataPartition.get(cvSet)), filename+"allID.csv");

//		InserterUtils.loadDelimitedData(data.getInserter(contrast, testDataPartition.get(cvSet)),
//				filename+"contrast_ids.csv");
//		InserterUtils.loadDelimitedData(data.getInserter(noncontrast, testDataPartition.get(cvSet)),
//				filename+"noncontrast_ids.csv");

		/*
		 * Load in the ground truth positive and negative segments
		 */
		InserterUtils.loadDelimitedData(data.getInserter(possentiment, trueTestDataPartition.get(cvSet)), 
			filename+"truepos.csv");

		InserterUtils.loadDelimitedData(data.getInserter(negsentiment, trueTestDataPartition.get(cvSet)), 
			filename+"trueneg.csv");


		Database trainDB = data.getDatabase(trainPartition.get(cvSet), [Prev,Priorpos,
			Priorneg,All,Tglpos,Tglneg] as Set);

//,Nrclexiconpos,Nrclexiconneg
		/*
		 * Setting the predicates possentiment and negsentiment to an initial value for all groundings
		 */
		List<Double> thresholdList = [0.5,0.45,0.4]
		ResultList allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(all))
		println "groundings for all"+ allGroundings.size();
		int j
		for (j = 0; j < allGroundings.size(); j++) {
			GroundTerm [] grounding = allGroundings.get(j)
			RandomVariableAtom atom1 = trainDB.getAtom(possentiment, grounding);
			RandomVariableAtom atom2 = trainDB.getAtom(negsentiment, grounding);
			atom1.setValue(0.0);
			atom2.setValue(0.0);
			atom1.commitToDB();
			atom2.commitToDB();
		}

		MPEInference inferenceApp = new MPEInference(m,trainDB, config)
		//LazyMPEInference inferenceApp = new LazyMPEInference(m, trainDB, config);
		inferenceApp.mpeInference();
		inferenceApp.close();
		println "trudatapartition : "+trueDataPartition.get(cvSet)
		Database trueDataDB = data.getDatabase(trueDataPartition.get(cvSet), [possentiment,negsentiment] as Set);
//		MaxLikelihoodMPE weightLearning = new MaxLikelihoodMPE(m, trainDB, trueDataDB, config);
//		MaxMargin weightLearning = new MaxMargin(m, trainDB, trueDataDB, config);
//		MaxMargin weightLearning = new MaxMargin(m, trainDB, trueDataDB, config);
		DualEM weightLearning = new DualEM(m, trainDB, trueDataDB, config);
//		LazyMaxLikelihoodMPE weightLearning = new LazyMaxLikelihoodMPE(m, trainDB, trueDataDB, config);
		//MaxPseudoLikelihood weightLearning = new MaxPseudoLikelihood(m, trainDB, trueDataDB, config);
		weightLearning.learn();
		weightLearning.close();
		/*
		 * Newly learned weights
		 */

		println m
		/*
		 */

		/*Test database setup*/

		Database testDB = data.getDatabase(testDataPartition.get(cvSet),
				[ Prev, Priorpos, Priorneg,All,Tglpos,Tglneg] as Set);
                                
                Database trueTestDB = data.getDatabase(trueTestDataPartition.get(cvSet), [possentiment, negsentiment] as Set);

		ResultList groundings = testDB.executeQuery(Queries.getQueryForAllAtoms(all))
		print groundings.size();
		for (j = 0; j < groundings.size(); j++) {
			GroundTerm [] grounding = groundings.get(j)
			RandomVariableAtom atom1 = testDB.getAtom(possentiment, grounding);
			RandomVariableAtom atom2 = testDB.getAtom(negsentiment, grounding);
			atom1.setValue(0.0);
			atom2.setValue(0.0);
			atom1.commitToDB();
			atom2.commitToDB();
		}
		inferenceApp = new MPEInference(m, testDB,config)
		inferenceApp.mpeInference();
		inferenceApp.close();




\
		//file4.append("Testfold" +"\t"+ "sentiment" +"\t"+ "AUPRC" +"\t"+ "NEGAUPRC"+"\t"+"AreaROC \n")

		def comparator = new SimpleRankingComparator(testDB)
		comparator.setBaseline(trueTestDB)

		// Choosing what metrics to report
		def metrics = [ RankingScore.AUPRC, RankingScore.NegAUPRC,  RankingScore.AreaROC]
		double [] score = new double[metrics.size()]
		double [] score2 = new double[metrics.size()]
		try {
			for (j = 0; j < metrics.size(); j++) {
				comparator.setRankingScore(metrics.get(j))
				score[j] = comparator.compare(possentiment)
				score2[j] = comparator.compare(negsentiment)
			}
		}
		catch (ArrayIndexOutOfBoundsException e) {
			System.out.println("No evaluation data! Terminating!");
		}

		Set<GroundAtom> groundings3 = Queries.getAllAtoms(trueTestDB, possentiment)
		int totalPosTestExamples3 = groundings3.size()

		groundings3 = Queries.getAllAtoms(trueTestDB, negsentiment)
		int totalNegTestExamples3 = groundings3.size()

		Set<GroundAtom> groundings2 = Queries.getAllAtoms(trueDataDB, possentiment)
		int totalPosTrainExamples = groundings2.size()

		groundings2 = Queries.getAllAtoms(trueDataDB, negsentiment)
		int totalNegTrainExamples = groundings2.size()

		int total =  totalNegTrainExamples+totalPosTestExamples3+totalNegTestExamples3+totalPosTrainExamples
		println "Total ###"+total
		println "Pos ###"+totalPosTrainExamples
		println "Ned ###"+totalNegTrainExamples

		println ( "Learned model:\n")
		println m

		/*
		 * Accuracy
		 */
                  
		Set<GroundAtom> groundings1 = Queries.getAllAtoms(trueTestDB, possentiment)
		int totalPosTestExamples = groundings1.size()
		println "printing totalTestExamples:Possentiment"+totalPosTestExamples
		groundings2 = Queries.getAllAtoms(trueTestDB, negsentiment)
		int totalNegTestExamples = groundings2.size()
		println "printing totalTestExamples: Negsentiment"+totalNegTestExamples


		//file3.append("\n scores for" +"\t"+"possentiment"+"\n")
		DiscretePredictionComparator poscomparator = new DiscretePredictionComparator(testDB)
		poscomparator.setBaseline(trueTestDB)
		poscomparator.setResultFilter(new MaxValueFilter(possentiment, 1))

		DiscretePredictionStatistics stats;
		Double accuracy = 0
		Double f1 = 0
		Double p = 0
		Double r = 0
		//file3.append("CVSet"+"\t"+"Pol"+"\t"+"Th"+"\t"+"Accuracy"+"\t"+"F1"+"\t"+"Precision"+"\t"+"Recall"+"\n")


		for(threshold in thresholdList)
		{

			//	file3.append("\n With threshold " +"\t"+threshold+"\n")
			poscomparator.setThreshold(threshold) // treat best value as true as long as it is nonzero

			stats = poscomparator.compare(possentiment, totalNegTestExamples+totalPosTestExamples)
			accuracy = stats.getAccuracy()
			f1 = stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE)
			p = stats.getPrecision(DiscretePredictionStatistics.BinaryClass.POSITIVE)
			r = stats.getRecall(DiscretePredictionStatistics.BinaryClass.POSITIVE)
			println (cvSet+"\t"+"pos"+"\t"+threshold+"\t"+accuracy+"\t"+f1+"\t"+p+"\t"+r+"\n")
		}
		DiscretePredictionComparator negcomparator = new DiscretePredictionComparator(testDB)
		negcomparator.setBaseline(trueTestDB)
		negcomparator.setResultFilter(new MaxValueFilter(negsentiment, 1))

		for (threshold in thresholdList)
		{
			//	file3.append("\n negsentiment with threshold =====" + threshold+"\n")

			negcomparator.setThreshold(threshold) // treat best value as true as long as it is nonzero
			stats = negcomparator.compare(negsentiment, totalNegTestExamples+totalPosTestExamples)
			accuracy = stats.getAccuracy()
			f1 = stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE)
			p = stats.getPrecision(DiscretePredictionStatistics.BinaryClass.POSITIVE)
			r = stats.getRecall(DiscretePredictionStatistics.BinaryClass.POSITIVE)
			println (cvSet+"\t"+"neg"+"\t"+threshold+"\t"+accuracy+"\t"+f1+"\t"+p+"\t"+r+"\n")
		}
		/*all_tn = negcomparator.tn+ poscomparator.tn;
		 all_tp = negcomparator.tp+ poscomparator.tp;
		 all_fp = negcomparator.fp+ poscomparator.fp;
		 all_fn = negcomparator.fn+ poscomparator.fn;
		 all_accuracy = (all_tn+all_tp)/(all_tn+all_tp+all_fp+all_fn)
		 println "Overall accuracy = " +all_accuracy*/



		trueDataDB.close();
		trainDB.close();
		testDB.close();
		trueTestDB.close();
	}



}