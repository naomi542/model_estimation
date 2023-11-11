function point_class = determine_class (AB, BC, AC)        
    if AB >= 0 && AC >= 0
        point_class= -1; % class A
    elseif BC >= 0 && AB <= 0 
        point_class= 0; % class B
    elseif AC <= 0 && BC <=0
        point_class= 1;  % class C
    end
end