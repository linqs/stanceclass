package edu.ucsc.cs.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
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

public class Evaluator {
	
	private final Database result;
	private Predicate p;
        private String name;
        private String fold;
	
	public Evaluator(Database result, Predicate p, String name, String fold) {
		this.result = result;
		this.p = p;
                this.name = name;
                this.fold = fold;		
	}
	
	public void outputToFile(){
		BufferedWriter writer = null;
		String dir = "output" + java.io.File.separator;
		String resultsFile = dir +  fold + "_result_" + name + ".csv";
                
		try {
			writer = new BufferedWriter(new FileWriter(resultsFile));
			
			for (GroundAtom atom : Queries.getAllAtoms(result, p)){
				GroundTerm[] terms = atom.getArguments();
                                
                                StringBuilder output = new StringBuilder();
                                for (GroundTerm t : terms){
                                    output.append(t + ",");
                                }
                                
				writer.append(output.toString() + atom.getValue() + "\n");
                                writer.flush();
			}
                } catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
