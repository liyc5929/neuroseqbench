function z = resizeZ(z_360,dim_output)

interval = ceil(360/dim_output);
z=zeros(1,dim_output);
count =1;
for angle = (0+interval):interval:360
    
    
    z(count) = z_360(angle);
    
    count=count+1;
    
end




end