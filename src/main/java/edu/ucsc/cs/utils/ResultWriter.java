package edu.ucsc.cs.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.evaluation.statistics.filter.AtomFilter;
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.ObservedAtom;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.util.database.Queries;

public class ResultWriter {
	
	private double[] scores;
        private String fold;
	private String fileName;
        
	public ResultWriter(double[] scores, String fold, String fileName){
            this.scores = scores;
            this.fold = fold;
            this.fileName = fileName;
        }
	
	public void write(){
		BufferedWriter writer = null;
		String dir = "output" + java.io.File.separator + "psl" + java.io.File.separator;
                
		try {
                        File resultsFile = new File(dir + fileName);
                        
//                        if(!resultsFile.exists()){
//                            resultsFile.createNewFile();
//                        }
                        resultsFile.createNewFile();
                    
			writer = new BufferedWriter(new FileWriter(resultsFile, true));
                        
                        StringBuilder output = new StringBuilder();
                                for (double score : scores){
                                    output.append(score + ",");
                                }
                                
                        writer.write(output.toString() + fold + "\n");
                        writer.flush();
                        
                        writer.close();
                } catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
