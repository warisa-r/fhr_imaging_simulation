import os

# To get the path of an xml file and its facet region file from msh assuming that
# dolfin-convert has been called
def msh2xml_path(msh_file_path):
    xml_path = msh_file_path.replace('.msh', '.xml')
    base, ext = os.path.splitext(xml_path)
    facet_region_xml_path = f"{base}_facet_region{ext}"
    return xml_path, facet_region_xml_path